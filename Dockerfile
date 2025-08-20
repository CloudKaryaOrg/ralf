FROM python:3.10-slim
# FROM python:3.12-slim  This new base docker image works
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /ralf

# Install system dependencies (if needed)
RUN apt-get update && rm -rf /var/lib/apt/lists/* && apt-get clean

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY ralf ./ralf

# Run Streamlit app
CMD ["python", "ralf/test_ralf.py"]