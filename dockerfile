FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg gfortran libopenblas-dev liblapack-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "mix_analyzer_web:app"]
