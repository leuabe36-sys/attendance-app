FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev

WORKDIR /app
COPY . .

RUN pip install flask werkzeug numpy opencv-python-headless face_recognition

CMD ["python", "main_beautiful_all_dashboards11111_2_.py"]
