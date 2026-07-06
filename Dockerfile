FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# No extra system dependencies needed for basic FastAPI

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 7860 (Hugging Face default)
EXPOSE 7860

# Command to run the FastAPI server (Supports Railway dynamic PORT)
CMD uvicorn api:app --host 0.0.0.0 --port ${PORT:-7860}
