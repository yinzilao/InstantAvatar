# conda python 3.8.1
# conda create -n instant python=3.8.1
pip install aitviewer==1.9.0
# pip install torch==1.13.1+cu116 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu116
# Update PyTorch and torchvision to use CUDA 11.8
# pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install fvcore iopath
pip install git+https://github.com/NVlabs/tiny-cuda-nn/@v1.6#subdirectory=bindings/torch
# pip install pytorch-lightning==1.5.7
# pip install opencv-python # reboot?
# pip install imageio
# pip install smplx==0.1.28
# pip install hydra-core==1.1.2
# pip install h5py ninja chumpy numpy==1.23.1
# pip install lpips


#########################################################
#########################################################
# install SAM (segment anything)
pip install git+https://github.com/facebookresearch/segment-anything.git
# download sam checkpoint
# https://github.com/facebookresearch/segment-anything#model-checkpoints
mkdir -p ./third_parties/segment-anything/ckpts
cd ./third_parties/segment-anything/ckpts
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
cd ../../../

###### install yolov8 - improve SAM accuracy using bounding boxes
pip install ultralytics

###### install yolov9
# Download source codes in third_parties
cd third_parties
# Clone the YOLOv9 repository
git clone https://github.com/carlosfab/yolov9.git
# Change the current working directory to the cloned YOLOv9 repository
cd yolov9
# Install the necessary YOLOv9 dependencies from the requirements.txt file
pip install -r requirements.txt -q

###############################################################
##################### sam2_yolov9 mamba env ####################
# OR, use yolov9 and SAM2 for better results
# need to create a new mamba env
mamba create -n sam2_yolov9 python=3.10 -y
mamba activate sam2_yolov9
# Install Required Packages for SAM2
# mamba install pytorch=2.3.1 torchvision=0.18.1 -c pytorch # no cuda support error
pip install torch>=2.3.1 torchvision>=0.18.1 --index-url https://download.pytorch.org/whl/cu118
# Install Required Packages for YOLOv9
apt update && apt upgrade -y
apt install zip htop screen libgl1-mesa-glx
pip install seaborn thop
# pip install ultralytics

# Download source codes in third_parties
cd third_parties
# Clone the YOLOv9 repository
git clone https://github.com/carlosfab/yolov9.git
# Change the current working directory to the cloned YOLOv9 repository
cd yolov9
# Install the necessary YOLOv9 dependencies from the requirements.txt file
pip install -r requirements.txt -q

# Back to third_parties folder
cd ..
# Clone the SAM2 repository and install it
git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .

# download sam2 checkpoint
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# back to root folder
cd ../../../..
# run sam2_yolov9
python scripts/custom/run-sam2_yolov9.py --data_dir data/custom/a2/

# back to base env
mamba deactivate

########################################################
########### SAM2 in Separate Docker Container ##########
docker compose build
docker compose up -d

# open in browser http://localhost:7262/ (SAM2 Web UI)
docker ps
#docker stop <container_id>
########################################################
########################################################


# install ROMP
pip install --upgrade "setuptools<60.0" 
pip install cython 
pip install lap
pip install simple_romp==1.1.3


####### install OpenPose #######
apt-get update && apt-get upgrade -y
apt-get remove --purge libboost-all-dev -y
apt-get install libboost-all-dev -y
apt-get install cmake-qt-gui -y
apt-get install libgflags-dev libgoogle-glog-dev libprotobuf-dev protobuf-compiler -y
apt-get install libopencv-dev -y # Add OpenCV
apt-get install libhdf5-dev -y
apt-get install libatlas-base-dev -y
apt-get install libgflags-dev -y

cd third_parties
git clone https://github.com/CMU-Perceptual-Computing-Lab/openpose.git
cd openpose

# Manually download models
mkdir -p models/pose/body_25/
mkdir -p models/face/
mkdir -p models/hand/

# Download model files manually from Dropbox mirrors
# Body model (BODY_25)
wget --no-check-certificate "https://www.dropbox.com/s/3x0xambj2rkyrap/pose_iter_584000.caffemodel?dl=1" -O models/pose/body_25/pose_iter_584000.caffemodel
# Face model
wget --no-check-certificate "https://www.dropbox.com/s/d08srojpvwnk252/pose_iter_116000.caffemodel?dl=1" -O models/face/pose_iter_116000.caffemodel
# Hand model
wget --no-check-certificate "https://www.dropbox.com/s/gqgsme6sgoo0zxf/pose_iter_102000.caffemodel?dl=1" -O models/hand/pose_iter_102000.caffemodel

git submodule update --init --recursive --remote
mkdir build/
cd build/
cmake-gui .. # cmake ..
make -j$(nproc)

find / -name "libopenpose.so.1.7.0" 2>/dev/null
export LD_LIBRARY_PATH=/app/third_parties/openpose/build/src/openpose:$LD_LIBRARY_PATH
# test openpose
build/examples/openpose/openpose.bin --video ../../data/custom/a2/a2.MP4 --face --hand --write_json ../../data/custom/a2/openpose_output


##########################
##########################
# docker build with retries
./setups/docker-build-retry.sh -f docker/Dockerfile -t iavatar:latest .
# Run docker image
# docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 ...

docker run --runtime=nvidia \
	-it --rm \
	--gpus=all \
    --shm-size=16g \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
	-e DISPLAY=$DISPLAY \
	-v /tmp/.X11-unix:/tmp/.X11-unix \
	-v "$(pwd)":/app \
	--network=host \
    --ipc=host \
    iavatar:latest

#########################################################
# Check NVIDIA driver
nvidia-smi

# Check CUDA version
nvcc --version

# Check if PyTorch sees CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

bash scripts/custom/run-openpose-bin-video.sh data/custom/a2/ a2.MP4

chmod +x ./scripts/custom/video-utils.py

bash scripts/custom/process-sequence-video.sh data/custom/a2/ a2.MP4

bash scripts/custom/process-sequence-video.sh data/custom/e1/ e1.MP4 female --force_rerun

#===============================================================

bash ./bash/run-semes-demo.sh 

python extract_mesh.py --config-name demo \
	dataset="semes/a2" \
	experiment="baseline" \
	deformer.opt.cano_pose="a_pose" \
	resume=false

python export_smpl_mesh.py --config-name SNARF_NGP_fitting dataset="semes/a2" experiment="baseline" deformer=smpl

pip install ultralytics
pip install mediapipe
pip install git+

VIDEO_FOLDER=data/custom/e1
GENDER=female
PREPROCESSED_IMAGE_FOLDER="preprocessed_images"
python scripts/custom/run-sam.py --data_dir $VIDEO_FOLDER --image_folder $PREPROCESSED_IMAGE_FOLDER

python scripts/custom/refine-smpl-large-errors.py     --data_dir $VIDEO_FOLDER     --gender $GENDER     --silhouette
