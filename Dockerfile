# 1. Use the official lightweight Python image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all our project files into the container
COPY . .

# 5. Expose the port FastAPI runs on
EXPOSE 8000

# 6. Command to run the API inside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]