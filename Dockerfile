# Базовый образ с предустановленным GDAL
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.8.4

WORKDIR /app

# Копируем только requirements, чтобы слои кешировались
COPY requirements.txt .

# Устанавливаем всё сразу:
#  - системные библиотеки для сборки (build-essential, git)
#  - libGL и прочие зависимости OpenCV
#  - pip‑зависимости
#  - очищаем кеши и убираем сборочные пакеты
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      build-essential \
      git \
      libgl1-mesa-glx \
      libglib2.0-0 \
      libsm6 \
      libxext6 \
      libxrender1 \
 && pip3 install --no-cache-dir -r requirements.txt \
 && apt-get purge -y --auto-remove build-essential git \
 && rm -rf /var/lib/apt/lists/*

# Копируем код
COPY . .

EXPOSE 5000
CMD ["python3", "app.py"]
