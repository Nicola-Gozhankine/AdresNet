"""
run_multiple_experiments.py

Запускает островную эволюцию несколько раз для заданной задачи (parity или 101)
и собирает статистику по лучшим результатам.
"""

import subprocess
import json
import os
import time

# Параметры
TASK = "101"                # "parity" или "101"
NUM_RUNS = 5                 # количество запусков
BASE_SEED = 42               # начальный seed (будет увеличиваться)
SCRIPT = "island_parity.py"  # используем тот же скрипт, но с изменённым импортом

# Создаём временную копию скрипта с нужным импортом
def prepare_script(task):
    with open("island_parity.py", "r") as f:
        content = f.read()
    if task == "parity":
        new_content = content.replace(
            "from fitness_pattern101 import fitness",
            "from fitness import fitness"
        )
    else:
        new_content = content.replace(
            "from fitness import fitness",
            "from fitness_pattern101 import fitness"
        )
    # Также заменим имя файла для сохранения результатов
    new_content = new_content.replace(
        "best_model.json",
        f"best_model_{task}.json"
    )
    temp_script = f"island_{task}_temp.py"
    with open(temp_script, "w") as f:
        f.write(new_content)
    return temp_script

def main():
    results = []
    temp_script = prepare_script(TASK)
    for run in range(NUM_RUNS):
        seed = BASE_SEED + run
        print(f"\n=== Запуск {run+1} для задачи {TASK} с seed {seed} ===")
        # Запускаем скрипт, передавая seed через переменную окружения
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)
        start = time.time()
        result = subprocess.run(
            ["python", temp_script],
            env=env,
            capture_output=True,
            text=True
        )
        duration = time.time() - start
        print(f"Время выполнения: {duration:.1f} сек")
        if result.returncode != 0:
            print("Ошибка:", result.stderr)
            continue
        # Извлекаем лучший raw из вывода
        lines = result.stdout.split('\n')
        best_raw = None
        for line in lines:
            if "Лучший фитнес (raw):" in line:
                best_raw = float(line.split(':')[1].strip())
                break
        if best_raw is not None:
            results.append(best_raw)
            print(f"Лучший raw: {best_raw:.4f}")
        else:
            print("Не удалось найти результат в выводе")

    # Статистика
    if results:
        avg = sum(results) / len(results)
        max_val = max(results)
        min_val = min(results)
        print(f"\n=== Результаты для задачи {TASK} ===")
        print(f"Запусков: {len(results)}")
        print(f"Среднее: {avg:.4f}")
        print(f"Макс: {max_val:.4f}")
        print(f"Мин: {min_val:.4f}")
        print(f"Разброс: {max_val - min_val:.4f}")
    else:
        print("Нет результатов")

    # Удаляем временный скрипт
    os.remove(temp_script)

if __name__ == "__main__":
    main()
    