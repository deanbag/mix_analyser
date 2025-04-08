FROM python:3.9-slim
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENV NUMBA_DISABLE_JIT=1  # Disable Numba JIT to avoid cffi hang
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "60", "mix_analyzer_web:app"]
