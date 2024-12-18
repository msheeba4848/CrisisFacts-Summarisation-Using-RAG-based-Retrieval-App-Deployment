# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    software-properties-common && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# Install PyTorch with MPS support and other Python dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the BERT model to avoid runtime downloads
RUN mkdir -p /app/models
RUN python3 -c "from transformers import AutoTokenizer, AutoModel; \
                AutoTokenizer.from_pretrained('bert-base-uncased', cache_dir='./models'); \
                AutoModel.from_pretrained('bert-base-uncased', cache_dir='./models')"

# Copy data and source code into the container
COPY ./data /app/data
COPY . .

# Expose the port
EXPOSE 5002

# Command to run the application
CMD ["python3", "api.py"]