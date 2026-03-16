#!/usr/bin/env python3
import subprocess
import time
import os
import sys
from datetime import datetime

def ensure_git_repo():
    """Проверяет, есть ли git-репозиторий, и создаёт его при необходимости."""
    try:
        subprocess.run(['git', 'status'], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("Git-репозиторий не найден. Создаю...")
        subprocess.run(['git', 'init'], check=True)
        subprocess.run(['git', 'add', '.'], check=True)
        subprocess.run(['git', 'commit', '-m', 'Initial commit'], check=True)
        print("Репозиторий создан.")

def git_commit():
    try:
        ensure_git_repo()
        subprocess.run(['git', 'add', '.'], check=True, capture_output=True)
        status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        if status.stdout.strip():
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            subprocess.run(['git', 'commit', '-m', f'Auto-commit at {timestamp}'], check=True, capture_output=True)
            print(f"[{timestamp}] Коммит создан.")
        else:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Нет изменений.")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка git: {e.stderr.decode() if e.stderr else str(e)}")

def main(interval=60):
    print(f"Запущен авто-коммит каждые {interval} секунд. Для остановки нажмите Ctrl+C.")
    try:
        while True:
            git_commit()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Авто-коммит остановлен.")

if __name__ == "__main__":
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    main(interval)