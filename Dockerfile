FROM python:3.9.6-slim

RUN apt-get update
RUN apt-get install -y \
    gcc \
    build-essential \
    ninja-build \
    libx11-dev \
    x11-utils \
    ffmpeg \
    libgtk-3-0 \
    libgstreamer-plugins-base1.0-0 \
    mpv \
    libmpv-dev \
    zenity \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app
COPY pyproject.toml ./
COPY requirements.lock ./
COPY README.md ./
RUN uv pip install --no-cache --system -r requirements.lock

COPY src .
RUN uv pip install --no-cache --system fluencyfeatureannotator/modules/resources/en_ud_L1L2e_combined_trf-0.0.1

# CMD python fluencyfeatureannotator/main.py
CMD flet run fluencyfeatureannotator -p 8001