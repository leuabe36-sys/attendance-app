FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libopenblas-dev \
    liblapack-dev \
    libx11-6 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --upgrade pip

RUN pip install \
    flask \
    werkzeug \
    numpy \
    opencv-python-headless

RUN pip install dlib==19.24.2 \
    --extra-index-url https://pypi.org/simple/ \
    --no-cache-dir

RUN pip install face_recognition

CMD ["python", "main_beautiful_all_dashboards11111_2_.py"]
