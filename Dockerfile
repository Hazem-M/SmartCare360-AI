FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# No extra system dependencies needed for basic FastAPI

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose port 8000 explicitly for Railway
EXPOSE 8000

# Command to run the FastAPI server on port 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
