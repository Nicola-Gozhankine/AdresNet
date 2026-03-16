import json
import sys

# Берем имя файла из аргументов командной строки
if len(sys.argv) < 2:
    print("Ошибка: укажи имя файла. Пример: python А4.py имя_файла.json")
    sys.exit(1)

file_name = sys.argv[1]

try:
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Сжимаем: убираем пробелы и отступы
    compact = json.dumps(data, separators=(',', ':'))
    print(compact)

except FileNotFoundError:
    print(f"Ошибка: файл {file_name} не найден.")
except Exception as e:
    print(f"Произошла ошибка: {e}")
