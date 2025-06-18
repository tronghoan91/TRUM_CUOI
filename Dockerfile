# Use official Python 3.10 image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port
EXPOSE 10000

# Run the application
CMD ["python", "main.py"]
