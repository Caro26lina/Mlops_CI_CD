# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (adjust if using FastAPI with uvicorn)
EXPOSE 5000

# Run the app (Flask or FastAPI)
CMD ["python", "app.py"]
