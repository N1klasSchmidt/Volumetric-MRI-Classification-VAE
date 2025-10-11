# Use CUDA 11.6 with conda base
FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create developer user with matching UID/GID from your system
RUN groupadd -g 1039 developer && \
    useradd -m -u 1038 -g 1039 -s /bin/bash developer

# Set up Python 3.10.8 in base environment
RUN conda install -y python=3.10.8 && \
    conda clean -afy

# Copy and install Python requirements
COPY requirements_clean.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Jupyter kernel for the base environment
RUN python -m ipykernel install --user --name=base --display-name "Python 3.10.8 (base)"

# Set ownership of workspace
RUN chown -R developer:developer /workspace

# Switch to developer user
USER developer

# Set environment variables for NVIDIA GPU
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["/bin/bash"]