import os
import re
import sys
import torch
import soundfile as sf
import numpy as np
from num2words import num2words

#3

# --- НАСТРОЙКИ ---
MODEL_PATH = "/Users/sarabeev/project/speech/silero_model.pt"
SPEAKER = "xenia"
SAMPLE_RATE = 48000
MAX_CHARS = 800
SPEED_FACTOR = 1.4
# ------------------


def get_chapter_number(chapter_text):
    """Извлекает только численный номер главы."""
    # Берем только первую строчку текста главы
    first_line = chapter_text.split('\n')[0]
    match = re.search(r'Глава\s+(\d+)', first_line)
    if match:
        return int(match.group(1))
    return 0



def replace_numbers_with_words(text):
    """Находит в тексте числа и заменяет их на пропись (1 -> один)."""
    # Функция находит все группы цифр
    def replace(match):
        number = int(match.group(0))
        # Конвертируем число в русский текст
        return num2words(number, lang='ru')
    
    # Заменяем регулярным выражением
    return re.sub(r'\d+', replace, text)

def parse_chapters_simply(full_text):
    """Надежный построчный сборщик глав без сложных регулярных выражений."""
    lines = full_text.splitlines()
    chapters = []
    current_chapter_lines = []
    
    for line in lines:
        # Проверяем, начинается ли строка со слова 'Глава' (игнорируя пробелы по краям)
        cleaned_line = line.strip()
        if cleaned_line.startswith("Глава "):
            # Если у нас уже накопился текст предыдущей главы — сохраняем её
            if current_chapter_lines:
                chapters.append("\n".join(current_chapter_lines))
            current_chapter_lines = [cleaned_line]  # Начинаем новую главу
        else:
            # Если это обычный текст, добавляем его к текущей главе
            if current_chapter_lines:
                current_chapter_lines.append(cleaned_line)
            else:
                # На случай, если в самом начале файла идет текст/вступление до "Главы 1"
                current_chapter_lines.append(cleaned_line)
                
    # Не забываем добавить последнюю главу
    if current_chapter_lines:
        chapters.append("\n".join(current_chapter_lines))
        
    return chapters

def get_chapter_name(chapter_text):
    """Извлекает номер из первой строки текста главы."""
    first_line = chapter_text.split('\n')[0]
    match = re.search(r'Глава\s+(\d+)', first_line)
    if match:
        # Превращаем строку с номером в число
        chapter_num = int(match.group(1))
        # Форматируем с 5 ведущими нулями
        return f"Chapter_{chapter_num:05d}"
    if match:
        return f"Chapter_{match.group(1)}"
    return "Chapter_Unknown"

def split_text_into_chunks(text, max_chars=800):
    """Режет текст главы на фрагменты до 800 символов."""
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

def main():
    if len(sys.argv) < 2:
        print("Ошибка: Не указано имя файла книги!")
        sys.exit(1)
        
    input_file = sys.argv[1]
    # Считываем номер начальной главы, если он передан (иначе начинаем с 0)
    start_from_chapter = 0
    if len(sys.argv) >= 3:
        try:
            start_from_chapter = int(sys.argv[2])
        except ValueError:
            print("Предупреждение: Номер главы должен быть числом. Начинаю с самого начала.")
    
    
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл текста '{input_file}' не найден!")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель '{MODEL_PATH}' не найдена!")
        sys.exit(1)

    print(f"Читаем файл: {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        full_text = f.read()

    # Парсим главы построчно
    all_chapters = parse_chapters_simply(full_text)
    
    # --- ФИЛЬТРАЦИЯ ГЛАВ ПО НАЧАЛЬНОМУ НОМЕРУ ---
    chapters_to_process = []
    for ch in all_chapters:
        ch_num = get_chapter_number(ch)
        if ch_num >= start_from_chapter:
            chapters_to_process.append((ch_num, ch))
            
    if not chapters_to_process:
        print(f"Главы с номером >= {start_from_chapter} не найдены в файле!")
        sys.exit(1)
        
    print(f"Всего глав в книге: {len(all_chapters)}")
    print(f"Будет озвучено глав (начиная с {start_from_chapter}): {len(chapters_to_process)}")


#    print(f"Обнаружено блоков/глав для озвучки: {len(chapters)}")

    print("Загрузка локальной модели...")
    device = torch.device('cpu')
    model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
    model.to(device)

    #for idx, chapter_content in enumerate(chapters_to_process, start=1):
    #    chapter_content = chapter_content.strip()
    for idx, (ch_num, chapter_content) in enumerate(chapters_to_process, start=1):
        chapter_content = chapter_content.strip()        
        if not chapter_content:
            continue
            
        base_name = get_chapter_name(chapter_content)
        output_wav = f"{base_name}.wav"
        
        chunks = split_text_into_chunks(chapter_content, MAX_CHARS)
        print(f"\n=== ПОДГОТОВКА [{idx}/{len(chapters_to_process)}] {base_name} ===")
        print(f"Глава разбита на {len(chunks)} частей. Итоговое имя: {output_wav}")
        
        chapter_audio_data = []
        
        try:
            for c_idx, chunk in enumerate(chunks, start=1):
                if c_idx == 1:
                    chunk = chunk.replace(" - ", ". ").replace(" – ", ". ")
                chunk = replace_numbers_with_words(chunk)
                # --- ТЕСТОВЫЙ ВЫВОД ТЕКСТА В КОНСОЛЬ ---
                print(f"--- Часть {c_idx}/{len(chunks)} отправляется в TTS: ---")
                print(chunk)  # Этот текст сейчас превратится в звук
                print("-" * 40)
                
                # Озвучка
                audio = model.apply_tts(
                    text=chunk,
                    speaker=SPEAKER,
                    sample_rate=SAMPLE_RATE
                )
                chapter_audio_data.append(audio.numpy())
            
            """
            # Сборка файла
            if chapter_audio_data:
                final_audio = np.concatenate(chapter_audio_data)
                sf.write(output_wav, final_audio, SAMPLE_RATE)
                print(f"Успешно сохранен файл: {output_wav}")
            """
            # Сборка файла
            if chapter_audio_data:
                final_audio = np.concatenate(chapter_audio_data)
                
                # --- ИЗМЕНЕНИЕ СКОРОСТИ ---
                if SPEED_FACTOR != 1.0:
                    """
                    # Вычисляем новую длину массива аудио
                    old_length = len(final_audio)
                    new_length = int(old_length / SPEED_FACTOR)
                    
                    # Интерполируем аудио-вектор для изменения скорости без искажения звука
                    final_audio = np.interp(
                        np.linspace(0, old_length, new_length),
                        np.arange(old_length),
                        final_audio
                    )
                    """
                    import audiotsm
                    import audiotsm.io.array

                    # Подготавливаем данные (конвертируем в формат, нужный для библиотеки)
                    # Silero выдает одномерный массив, делаем его двухмерным [каналы, отсчеты]
                    input_buffer = np.array([final_audio], dtype=np.float32)
                    reader = audiotsm.io.array.ArrayReader(input_buffer)

                    # Создаем пустой буфер для ускоренного аудио
                    output_buffer = audiotsm.io.array.ArrayWriter(1)

                    # Применяем алгоритм WSOLA (сохраняет тон голоса идеальным)
                    tsm = audiotsm.wsola(1, speed=SPEED_FACTOR)
                    tsm.run(reader, output_buffer)

                    # Забираем готовое ускоренное аудио (возвращаем в одномерный массив)
                    final_audio = output_buffer.data[0]
                # --------------------------
                
                sf.write(output_wav, final_audio, SAMPLE_RATE)
                print(f"Успешно сохранен файл (Скорость: {SPEED_FACTOR}x): {output_wav}")
                 # --- АВТОМАТИЧЕСКАЯ КОНВЕРТАЦИЯ В MP3 ЧЕРЕЗ FFMPEG ---
                output_mp3 = output_wav.replace(".wav", ".mp3")
                print(f"Конвертирую {output_wav} в MP3...")

                # Запускаем ffmpeg (флаг -y перезаписывает файл, если он существует)
                # -b:a 128k — отличный битрейт для аудиокниг (баланс веса и качества)
                ffmpeg_cmd = f'ffmpeg -y -i "{output_wav}" -b:a 128k "{output_mp3}" -loglevel error'

                # Выполняем команду в системе
                exit_code = os.system(ffmpeg_cmd)

                if exit_code == 0:
                    print(f"Успешно сохранен MP3: {output_mp3}")
                    os.remove(output_wav)  # Удаляем временный WAV, чтобы не занимал место
                else:
                    print(f"Предупреждение: FFmpeg вернул ошибку. Оставлен файл {output_wav}")
                # -----------------------------------------------------

                
                
        except Exception as e:
            print(f"Ошибка при обработке {base_name}: {e}")

    print("\nПроцесс завершен!")

if __name__ == "__main__":
    main()

