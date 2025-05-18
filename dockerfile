FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1 ffmpeg gfortran libopenblas-dev liblapack-dev
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PORT=8080
CMD ["sh", "-c", "gunicorn -w 2 -b 0.0.0.0:$PORT mix_analyzer_web:app"]
