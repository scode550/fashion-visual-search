# Dockerfile

FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .
# Copy FAISS index and metadata
COPY data/faiss_index.bin data/faiss_index.bin
COPY data/metadata_balanced.npy data/metadata_balanced.npy


# Download model weights (cached after first run)
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
               CLIPModel.from_pretrained('patrickjohncyh/fashion-clip'); \
               CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')"

# Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
