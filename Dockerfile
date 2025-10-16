# 1. Start with a lean and official Python base image
FROM python:3.10-slim

RUN apt-get update && apt-get install -y libpq-dev && rm -rf /var/lib/apt/lists/*

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy only the requirements file first to leverage Docker's caching
COPY requirements.txt .

# 4. Install all Python dependencies from requirements.txt
# This includes the extra PyTorch URL your file specifies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Now, copy your application code into the container
# This assumes your code is in a folder named 'app'
COPY main.py .

# 6. Define the command to run your application using Gunicorn
# This command points to the 'app' object inside your 'main.py' file within the 'app' module.
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--timeout", "0", "app.main:app"]