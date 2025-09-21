# Базовый образ с Python
FROM python:3.9-slim

# Установка системных зависимостей для ML-библиотек
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Установка рабочей директории
WORKDIR /app

# Копирование файлов с зависимостями
COPY requirements.txt .

# Установка Python-зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копирование всего кода проекта
COPY . .

# Команда по умолчанию (запуск main.py)
CMD ["python", "main.py"]