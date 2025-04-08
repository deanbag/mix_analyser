# Use a slim Python 3.9 base image to keep the container lightweight
FROM python:3.9-slim

# Install ffmpeg, required by matchering for audio processing
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy all project files to the container
COPY . .

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Disable Numba JIT to prevent compilation hangs
ENV NUMBA_DISABLE_JIT=1

# Expose port 8000 for Render
EXPOSE 8000

# Run the Flask app with Gunicorn, increased timeout to 60 seconds
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--timeout", "60", "mix_analyzer_web:app"]
