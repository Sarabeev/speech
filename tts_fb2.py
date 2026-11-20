#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Озвучка книг из FB2-файлов, лежащих внутри ZIP-архивов.

Скрипт, созданный на базе tts_book.py для книги HeavyKnight.txt.

Что делает:
  - Проходит по папке с ZIP-архивами (имена начинаются с двух цифр, e.g. "01 ... .fb2.zip").
  - Для каждого архива распаковывает единственный FB2 в память (на диск не пишет).
  - Разбирает FB2 по главам: каждая верхнеуровневая <section> в <body> — это одна глава.
  - Для каждой книги создаёт отдельную папку в выходном каталоге.
  - Каждую главу озвучивает в отдельный MP3-файл (переиспользует пайплайн Silero
    из tts_book.py: разбиение на куски, проговаривание чисел, ускорение, ffmpeg).

Использование:
    python3 tts_fb2.py
"""

import os
import re
import sys
import shutil
import zipfile
import xml.etree.ElementTree as ET

import torch
import soundfile as sf
import numpy as np
from num2words import num2words

# --- НАСТРОЙКИ ---
MODEL_PATH = "/Users/sarabeev/project/speech/silero_model.pt"
SPEAKER = "xenia"
SAMPLE_RATE = 48000
MAX_CHARS = 800
SPEED_FACTOR = 1.4

# Папка, в которой лежат ZIP-архивы с FB2 (файлы начинаются с двух цифр).
INPUT_DIR = "/Users/sarabeev/Downloads/ХудожественныеКниги/книги от Малюкова/Идеальный мир для Лекаря"
# Выходной каталог — для каждой книги создаётся своя подпапка.
OUTPUT_DIR = "/Users/sarabeev/project/speech"

# Пространство имён FB2.
FB2_NS = {"fb": "http://www.gribuser.ru/xml/fictionbook/2.0"}


# ------------------ Числа -> пропись ------------------
def replace_numbers_with_words(text):
    """Находит в тексте числа и заменяет их на пропись (1 -> один)."""
    def replace(match):
        number = int(match.group(0))
        return num2words(number, lang='ru')
    return re.sub(r'\d+', replace, text)


# ------------------ Разбиение на куски ------------------
def split_text_into_chunks(text, max_chars=800):
    """Режет текст главы на фрагменты до max_chars символов."""
    lines = text.split('\n')

    chunks = []
    current_chunk = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if len(line) > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sentence in sentences:
                if len(current_chunk) + len(sentence) + 1 > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += " " + sentence
        elif len(current_chunk) + len(line) + 1 > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += "\n" + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# ------------------ FB2 -> главы ------------------
def element_text(el, join_with=""):
    """Собирает текст внутри элемента, раскрывая вложенные теги.

    Теги <a> с текстом-ссылкой (http...) пропускаются — URL вслух не читаем.
    Остальной текст собирается рекурсивно.
    """
    XLINK = "{http://www.w3.org/1999/xlink}"
    parts = []
    for node in el.iter():
        tag = node.tag.split('}')[-1] if isinstance(node.tag, str) else ''
        if tag == 'a':
            # Ссылку-URL пропускаем полностью.
            href = node.get(XLINK + 'href', '') or ''
            if href.lower().startswith('http'):
                continue
        if node.text:
            # Неразрывные пробелы -> обычные (чище для прочтения).
            parts.append(node.text.replace('\xa0', ' '))
    return "".join(parts)


def get_para_lines(section_el):
    """Возвращает список строк (абзацев) главы.

    Обрабатываем блоковые элементы <p> и <subtitle>. Пустые строки и
    разделители * * * пропускаем.
    """
    lines = []
    for child in section_el:
        tag = child.tag.split('}')[-1] if isinstance(child.tag, str) else ''
        if tag not in ('p', 'subtitle'):
            continue
        text = element_text(child).strip()
        if not text:
            continue
        # Разделитель сцены "* * *" не озвучиваем.
        if re.fullmatch(r'[\s*—–-]+', text):
            continue
        lines.append(text)
    return lines


def parse_fb2_chapters(fb2_text):
    """Разбирает FB2 на главы.

    Возвращает список словарей: {'title': str, 'text': str}.

    Логика глав:
      - Главы — это верхнеуровневые <section> внутри <body>.
      - Вложенные <section> (под-главы) сливаются со своей родительской
        главой: их заголовки и текст добавляются в общий текст.
      - Первый <title> внутри <body> (название книги) пропускается.

    Нумерация глав своя — по порядку появления (начиная с 1), как в tts_book.py.
    """
    # Аннотация/description находится до <body>; берём только <body>.
    try:
        root = ET.fromstring(fb2_text)
    except ET.ParseError as e:
        raise ValueError(f"Не удалось разобрать FB2 как XML: {e}")

    body = root.find('fb:body', FB2_NS)
    if body is None:
        raise ValueError("В FB2 не найден <body>")

    chapters = []

    # Перебираем дочерние элементы <body>: пропускаем первый <title> (имя книги).
    for section_el in body:
        tag = section_el.tag.split('}')[-1] if isinstance(section_el.tag, str) else ''
        if tag != 'section':
            continue  # title, empty-line и т.п. пропускаем

        # Заголовок главы — первый <title> верхнего уровня.
        title_el = section_el.find('fb:title', FB2_NS)
        chapter_title = element_text(title_el).strip() if title_el is not None else ""

        # Собираем текст: все блоки <p>/<subtitle> верхнего уровня плюс
        # под-разделы (вложенные <section>), рекурсивно.
        lines = []
        for child in section_el:
            tag = child.tag.split('}')[-1] if isinstance(child.tag, str) else ''
            if tag == 'title':
                # Заголовок верхнего уровня уже взяли; заголовки вложенных
                # под-разделов добавляем отдельной строкой.
                text = element_text(child).strip()
                if text and text != chapter_title:
                    lines.append(text)
            elif tag in ('p', 'subtitle'):
                text = element_text(child).strip()
                if not text:
                    continue
                if re.fullmatch(r'[\s*—–-]+', text):
                    continue
                lines.append(text)
            elif tag == 'section':
                # Вложенный под-раздел: сливаем его строки в текущую главу.
                sub_lines = get_para_lines(child)
                lines.extend(sub_lines)

        chapter_text = "\n".join(lines).strip()
        if not chapter_text:
            continue

        chapters.append({
            "title": chapter_title,
            "text": chapter_text,
        })

    return chapters


# ------------------ Имена ------------------
def make_chapter_filename(index):
    """Имя файла главы по порядковому номеру (собственная нумерация от 1)."""
    return f"Chapter_{index:05d}.mp3"


def make_book_folder_name(zip_filename):
    """Имя папки книги из имени архива.

    "01 Идеальный мир для Лекаря.fb2.zip" -> "Идеальный мир для Лекаря"
    "02 Идеальный мир для Лекаря 2.fb2.zip" -> "Идеальный мир для Лекаря 2"
    """
    name = os.path.basename(zip_filename)
    # Убираем расширение .fb2.zip
    base = re.sub(r'\.fb2\.zip$', '', name, flags=re.IGNORECASE)
    # Убираем ведущие две цифры и пробел
    base = re.sub(r'^\d{1,2}\s+', '', base).strip()
    return base or "Книга"


def sanitize_folder_name(name):
    """Делает имя безопасным для имени папки."""
    return re.sub(r'[\\/:*?"<>|]', '_', name).strip()


# ------------------ Озвучка ------------------
def tts_chapter(model, index, title, text, out_mp3):
    """Озвучивает одну главу в MP3.

    В начале каждой главы произносится её номер и, если есть, название
    (например «Глава три. Название главы»).
    """
    # Произносимый номер главы прописью: 1 -> "глава один", 2 -> "глава два"...
    number_words = num2words(index, lang='ru')

    # Если в названии главы уже есть слово "Глава" с номером (обычный случай
    # "Глава 1", "Глава 2", ...), озвучиваем само название — это уже и есть
    # «глава + номер». Цифры в названии проговариваем прописью.
    m_heading = re.match(r'^\s*Глава\s+\d+\s*$', title or '', flags=re.IGNORECASE)
    if m_heading:
        intro = replace_numbers_with_words(title)
    else:
        # Иначе озвучиваем "Глава <номер>.<название, если есть>".
        intro = f"Глава {number_words}."
        if title:
            intro += f" {title}"

    chunks = split_text_into_chunks(text, MAX_CHARS)
    print(f"  Глава разбита на {len(chunks)} частей")

    chapter_audio_data = []
    # Сначала озвучиваем заголовок: "Глава N. Название".
    audio = model.apply_tts(
        text=intro,
        speaker=SPEAKER,
        sample_rate=SAMPLE_RATE
    )
    chapter_audio_data.append(audio.numpy())
    print(f"  --- Заголовок главы -> TTS ---")
    print("  " + intro)
    print("-" * 44)

    for c_idx, chunk in enumerate(chunks, start=1):
        if c_idx == 1:
            chunk = chunk.replace(" - ", ". ").replace(" – ", ". ")
        chunk = replace_numbers_with_words(chunk)
        print(f"  --- Часть {c_idx}/{len(chunks)} -> TTS ---")
        print("  " + chunk[:200].replace("\n", " "))
        print("-" * 44)

        audio = model.apply_tts(
            text=chunk,
            speaker=SPEAKER,
            sample_rate=SAMPLE_RATE
        )
        chapter_audio_data.append(audio.numpy())

    if not chapter_audio_data:
        print("  Нет аудио, глава пропущена")
        return False

    final_audio = np.concatenate(chapter_audio_data)

    # Ускорение (WSOLA — сохраняет тон голоса)
    if SPEED_FACTOR != 1.0:
        import audiotsm
        import audiotsm.io.array
        input_buffer = np.array([final_audio], dtype=np.float32)
        reader = audiotsm.io.array.ArrayReader(input_buffer)
        output_buffer = audiotsm.io.array.ArrayWriter(1)
        tsm = audiotsm.wsola(1, speed=SPEED_FACTOR)
        tsm.run(reader, output_buffer)
        final_audio = output_buffer.data[0]

    output_wav = out_mp3.replace(".mp3", ".wav")
    sf.write(output_wav, final_audio, SAMPLE_RATE)

    ffmpeg_cmd = f'ffmpeg -y -i "{output_wav}" -b:a 128k "{out_mp3}" -loglevel error'
    exit_code = os.system(ffmpeg_cmd)
    if exit_code == 0:
        os.remove(output_wav)
        return True
    print(f"  Предупреждение: ffmpeg вернул ошибку, оставлен {output_wav}")
    return False


def process_zip(model, zip_path, out_dir):
    """Разбирает и озвучивает одну книгу (один ZIP)."""
    print("=" * 70)
    print(f"Книга: {os.path.basename(zip_path)}")
    print("=" * 70)

    # Читаем FB2 из ZIP в память.
    fb2_text = None
    with zipfile.ZipFile(zip_path, 'r') as zf:
        fb2_names = [n for n in zf.namelist() if n.lower().endswith('.fb2')]
        if not fb2_names:
            print("  Ошибка: в ZIP нет FB2-файла, пропускаю")
            return
        fb2_text = zf.read(fb2_names[0]).decode('utf-8', errors='replace')

    try:
        chapters = parse_fb2_chapters(fb2_text)
    except ValueError as e:
        print(f"  Ошибка разбора FB2: {e}")
        return

    if not chapters:
        print("  В книге не найдено ни одной главы, пропускаю")
        return

    print(f"  Найдено глав: {len(chapters)}")
    os.makedirs(out_dir, exist_ok=True)

    for idx, chapter in enumerate(chapters, start=1):
        title_disp = chapter["title"] or "(без заголовка)"
        out_mp3 = os.path.join(out_dir, make_chapter_filename(idx))
        print(f"\n  === ГЛАВА {idx}/{len(chapters)} [{title_disp}] === -> {os.path.basename(out_mp3)}")
        try:
            ok = tts_chapter(model, idx, chapter["title"], chapter["text"], out_mp3)
            if ok:
                print(f"  Сохранено: {out_mp3}")
        except Exception as e:
            print(f"  Ошибка при обработке главы {idx} ({title_disp}): {e}")

    print(f"\nКнига обработана. Результат в: {out_dir}")


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель '{MODEL_PATH}' не найдена!")
        sys.exit(1)
    if not os.path.isdir(INPUT_DIR):
        print(f"Ошибка: Папка с книгами '{INPUT_DIR}' не найдена!")
        sys.exit(1)

    # Собираем ZIP-архивы, имена которых начинаются с двух цифр.
    zip_files = []
    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith('.zip'):
            continue
        base = os.path.splitext(fname)[0]
        if re.match(r'^\d{1,2}', base):
            zip_files.append(os.path.join(INPUT_DIR, fname))

    if not zip_files:
        print("Не найдено ни одного ZIP-файла, начинающегося с цифр!")
        sys.exit(1)

    zip_files.sort()
    print(f"Найдено книг (ZIP): {len(zip_files)}")
    print("Загрузка локальной модели...")
    device = torch.device('cpu')
    model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
    model.to(device)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for zip_path in zip_files:
        folder = sanitize_folder_name(make_book_folder_name(zip_path))
        out_dir = os.path.join(OUTPUT_DIR, folder)
        process_zip(model, zip_path, out_dir)

    print("\nПроцесс завершен!")


if __name__ == "__main__":
    main()
