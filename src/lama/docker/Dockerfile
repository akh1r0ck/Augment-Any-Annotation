FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget mc tmux nano build-essential rsync libgl1 python3 python3-pip

ARG USERNAME=user
RUN apt-get install -y sudo && \
    addgroup --gid 1000 $USERNAME && \
    adduser --uid 1000 --gid 1000 --disabled-password --gecos '' $USERNAME && \
    adduser $USERNAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    USER=$USERNAME && \
    GROUP=$USERNAME

USER $USERNAME:$USERNAME
WORKDIR "/home/$USERNAME"
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV PYTHONPATH="/home/$USERNAME/project"

RUN python3 -m pip install --user -U pip
RUN python3 -m pip install --user torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
RUN python3 -m pip install --user numpy~=1.26.0 scipy~=1.13.0 opencv-python~=4.10.0 opencv-python-headless~=4.10.0 tensorflow joblib matplotlib pandas \
    albumentations==0.5.2 pytorch-lightning~=2.6.1 tabulate easydict==1.9.0 kornia==0.5.0 webdataset \
    packaging gpustat tqdm pyyaml hydra-core~=1.3.2 scikit-learn==1.5.0 tensorboard tabulate
RUN python3 -m pip install --user scikit-image~=0.24.0

ENV TORCH_HOME="/home/$USERNAME/.torch"

ADD entrypoint.sh /home/$USERNAME/.local/bin/entrypoint.sh
ENTRYPOINT [ "entrypoint.sh" ]
