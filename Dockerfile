# Use the official NVIDIA CUDA runtime as the foundation
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install core system utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda into a global directory
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# Add conda to the global system path
ENV PATH=$CONDA_DIR/bin:$PATH

# Create the environment and pre-install astra-toolbox via conda channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create --name ndtomo python=3.11 -y && \
    conda install -n ndtomo -c astra-toolbox -c nvidia astra-toolbox -y && \
    conda clean -afy

# Ensure all subsequent RUN commands execute inside the active conda environment shell
SHELL ["conda", "run", "-n", "ndtomo", "/bin/bash", "-c"]

# Install your pip dependencies, pinned setuptools fix, and PyTorch CUDA modules
RUN python -m pip install --upgrade pip && \
    python -m pip install notebook nbconvert ipykernel "setuptools<81" && \
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and install nDTomo directly from your source repository
WORKDIR /app
COPY . /app
RUN python -m pip install .

# Configure the container to launch an internal Jupyter notebook server on startup
EXPOSE 8888
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "ndtomo", "jupyter", "notebook", "--ip=0.0.0.0", "--no-browser", "--allow-root"]