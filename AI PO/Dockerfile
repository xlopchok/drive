# Базовый образ для сборки
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 AS builder

# Устанавливаем рабочую директорию
WORKDIR /AI_PO

# Копируем requirements.txt заранее, чтобы Docker мог кэшировать установку зависимостей
COPY ./requirements.txt .

# Устанавливаем Python, pip и необходимые утилиты
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python3.9 python3.9-distutils git wget curl unzip && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    python3.9 -m pip install --upgrade pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Устанавливаем зависимости из requirements.txt
RUN python3.9 -m pip install -r requirements.txt

# Обновляем numpy до указанной версии
RUN python3.9 -m pip install -U numpy==1.26.4

# Копируем скрипт для загрузки моделей
COPY ./download_models.py .

# Выполняем скрипт для скачивания модели
RUN python3.9 download_models.py

# Создаём директорию для моделей
RUN mkdir -p /AI_PO/model

# Второй этап сборки: компиляция
FROM builder AS stage

# Устанавливаем дополнительные библиотеки для компиляции
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libopencv-dev \
    libeigen3-dev \
    libassimp-dev \
    cmake \
    && apt-get clean

# Копируем исходный файл C++ для компиляции
COPY mesh_projection_mt.cpp .

# Компилируем C++ код
RUN g++ -std=c++17 -O2 -o mesh_projection_mt mesh_projection_mt.cpp \
    -I/usr/include/opencv4 \
    -I/usr/include/eigen3 \
    -lopencv_core \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lassimp \
    -pthread

# Устанавливаем рабочую директорию
WORKDIR /AI_PO

# Копируем оставшуюся часть приложения
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Команда запуска приложения
CMD ["streamlit", "run", "front_inference.py"]
