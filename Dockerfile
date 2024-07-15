ARG tag=1.13.1-cuda11.6-cudnn8-runtime

# Base image
FROM pytorch/pytorch:${tag}

LABEL maintainer='Azzam'
LABEL version='0.0.1'


# Install Ubuntu packages
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        git \
        libgl1\
        curl \
        wget \
        libglib2.0-0\
    && rm -rf /var/lib/apt/lists/*

# Update python packages
RUN python3 --version && \
    pip3 install --no-cache-dir --upgrade pip "setuptools<60.0.0" wheel

# Set LANG environment
ENV LANG=C.UTF-8

# Set the working directory
WORKDIR /data

RUN pip3 install --no-cache-dir jupyterlab ultralytics supervision 'git+https://github.com/facebookresearch/segment-anything.git'

RUN git clone https://github.com/SkalskiP/yolov9.git && \
    pip3 install --no-cache-dir -r yolov9/requirements.txt

RUN mkdir -p /data/models && \
    wget -P models https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && \
    wget -P yolov9/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt && \
    wget -P yolov9/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-e.pt && \
    wget -P yolov9/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt && \
    wget -P yolov9/weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-e.pt

# Install the specific version of scikit-learn
RUN pip3 install --no-cache-dir scikit-learn==1.2.2

COPY rf_model.joblib /data/models/rf_model.joblib

CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=\"argon2:$argon2id$v=19$m=10240,t=10,p=8$X1hRpZbS3gQ09WAawc/rwg$NZoj3vphgpAUrkwNC7c2OLeiKmspbBgdKulmbiVr2UE\"", "--notebook-dir='/data'", "--NotebookApp.allow_origin='*'"]
