# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYOPENGL_PLATFORM=egl
ENV CONDA_ENV_NAME=iavatar-env

# Make CUDA visible to PyTorch
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Add NVIDIA runtime environment variables
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9"

# Install system dependencies
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4 && \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    apt-get update --fix-missing && apt-get upgrade -y \
    && apt-get install -y --fix-missing --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    # Dependencies for mesh package
    libboost-dev \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    # OpenGL and display dependencies
    ffmpeg \
    libegl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge3
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh \
    && bash miniforge.sh -b -p /opt/conda \
    && rm miniforge.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Create conda environment with Python 3.8
RUN mamba create -n ${CONDA_ENV_NAME} python=3.8 -y

# Set up conda environment activation
SHELL ["mamba", "run", "-n", "iavatar-env", "/bin/bash", "-c"]

# Set CUDA architecture for tiny-cuda-nn, should be aligned with TORCH_CUDA_ARCH_LIST
ENV TCNN_CUDA_ARCHITECTURES="75;80;86;89"
# Install PyTorch 2.0.0 with CUDA 11.8
RUN pip install torch==2.0.0 torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip install "pytorch-lightning<=1.8.1" --index-url https://pypi.org/simple
RUN pip install git+https://github.com/NVlabs/tiny-cuda-nn/@v1.6#subdirectory=bindings/torch



RUN pip install aitviewer==1.9.0 \
    && pip install fvcore iopath \
    && pip install opencv-python \
    && pip install imageio \
    && pip install smplx==0.1.28 \
    && pip install hydra-core==1.1.2 \
    && pip install h5py ninja chumpy numpy==1.23.1 \
    && pip install lpips

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app/

####### install 3rd party tools: Segment Anything, YOLOv9, ROMP #######
####### install SAM (segment anything) #######
RUN pip install git+https://github.com/facebookresearch/segment-anything.git
# download sam checkpoint
# https://github.com/facebookresearch/segment-anything#model-checkpoints
RUN mkdir -p ./third_parties/segment-anything/ckpts
RUN if [ ! -f ./third_parties/segment-anything/ckpts/sam_vit_h_4b8939.pth ]; then \
    cd ./third_parties/segment-anything/ckpts && \
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    cd ../../../; \
fi

####### install ROMP #######
RUN pip install --upgrade "setuptools<60.0" 
RUN pip install cython 
RUN pip install lap
RUN pip install simple_romp==1.1.3
# Install PyTorch3D with specific version that matches PyTorch 2.0.0
RUN pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu118_pyt200/download.html

# Install nvdiffrast properly for GPU support
RUN pip install ninja
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git

# Install Kaolin and other dependencies
RUN pip install pygltflib==1.15.0 \
    ipyevents==2.0.1 \
    ipycanvas==0.9.1 \
    "jupyter-client<8.0.0" \
    flask \
    pybind11 \
    "pyzmq<25.0.0" \
    tornado \
    usd-core==23.5 \
    && pip install --no-deps kaolin==0.14.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html

# Modify chumpy to fix numpy import issues
RUN sed -i 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import bool_ as bool, int_ as int, float_ as float, complex_ as complex, object_ as object, str_ as str, nan, inf/' /opt/conda/envs/iavatar-env/lib/python3.8/site-packages/chumpy/__init__.py

# Create directory and download ROMP model only if it doesn't exist
RUN mkdir -p /root/.romp/
RUN if [ ! -f /root/.romp/ROMP.pkl ]; then \
    wget -O /root/.romp/ROMP.pkl https://github.com/Arthur151/ROMP/releases/download/V2.0/ROMP.pkl || \
    echo "Manual download required: place ROMP.pkl in /root/.romp/"; \
fi

RUN romp.prepare_smpl -source_dir=./third_parties/romp/smpl_model_data/

####### install OpenPose #######
RUN apt-get update && apt-get upgrade -y
RUN apt-get remove --purge libboost-all-dev -y
RUN apt-get install libboost-all-dev -y
RUN apt-get install cmake-qt-gui -y
RUN apt-get install libgflags-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler -y
RUN apt-get install libopencv-dev -y # Add OpenCV
RUN apt-get install libhdf5-dev -y
RUN apt-get install libatlas-base-dev -y
RUN apt-get install libgflags-dev -y

RUN if [ ! -d ./third_parties/openpose ]; then \
    cd third_parties \
    && git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git \
    && cd openpose \
    && git submodule update --init --recursive --remote; \
fi

# Create model directories
RUN mkdir -p third_parties/openpose/models/pose/body_25/ \
    && mkdir -p third_parties/openpose/models/face/ \
    && mkdir -p third_parties/openpose/models/hand/

# Download model files if they don't exist
RUN if [ ! -f third_parties/openpose/models/pose/body_25/pose_iter_584000.caffemodel ]; then \
    wget --no-check-certificate "https://www.dropbox.com/s/3x0xambj2rkyrap/pose_iter_584000.caffemodel?dl=1" \
    -O third_parties/openpose/models/pose/body_25/pose_iter_584000.caffemodel; \
fi

RUN if [ ! -f third_parties/openpose/models/face/pose_iter_116000.caffemodel ]; then \
    wget --no-check-certificate "https://www.dropbox.com/s/d08srojpvwnk252/pose_iter_116000.caffemodel?dl=1" \
    -O third_parties/openpose/models/face/pose_iter_116000.caffemodel; \
fi

RUN if [ ! -f third_parties/openpose/models/hand/pose_iter_102000.caffemodel ]; then \
    wget --no-check-certificate "https://www.dropbox.com/s/gqgsme6sgoo0zxf/pose_iter_102000.caffemodel?dl=1" \
    -O third_parties/openpose/models/hand/pose_iter_102000.caffemodel; \
fi

RUN mkdir -p third_parties/openpose/build/
RUN cd third_parties/openpose/build/ && cmake .. \
    && make -j$(nproc)

# Add OpenPose library path to system library config
ENV LD_LIBRARY_PATH=/app/third_parties/openpose/build/src/openpose:$LD_LIBRARY_PATH
RUN echo "/app/third_parties/openpose/build/src/openpose" > /etc/ld.so.conf.d/openpose.conf && \
    ldconfig

###### setup environment for yolov9 #######
# Download source codes in third_parties
RUN if [ ! -d ./third_parties/yolov9 ]; then \
    cd third_parties \
    && git clone https://github.com/carlosfab/yolov9.git \
    && cd yolov9; \
fi
RUN cd third_parties/yolov9 \
    && pip install -r requirements.txt -q

# Install Ultralytics for YoLov8
RUN pip install ultralytics

###### Models for aitviewer #######
RUN mkdir -p ../data/smplx_models
RUN if [ ! -L /data/smplx_models/smpl ]; then \
    ln -s /app/data/SMPLX/smpl/ /data/smplx_models/; \
fi

###### Models for instant avatar #######
RUN mkdir -p /root/.cache/torch/hub/checkpoints/ && \
    if [ ! -f /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth ]; then \
        curl -L https://download.pytorch.org/models/vgg16-397923af.pth -o /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth; \
    fi
RUN chmod 644 /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth

#### END setup environment for 3rd party tools #######
######################################################
ENV HYDRA_FULL_ERROR=1

# Initialize mamba in bash and set up environment activation
RUN mamba init bash && \
    echo "mamba activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Set default command to start bash with mamba environment activated
SHELL ["/bin/bash", "--login", "-c"]
ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "iavatar-env", "/bin/bash"]
