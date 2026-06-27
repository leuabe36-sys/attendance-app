FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip
RUN pip install cmake
RUN pip install dlib --no-cache-dir
RUN pip install flask werkzeug numpy opencv-python-headless face_recognition

CMD ["python", "main_beautiful_all_dashboards11111_2_.py"]
