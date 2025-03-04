# select Image
FROM nvidia/cuda:11.3.1-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

RUN apt-get update
RUN apt-get install -y libicu-dev git wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 mercurial subversion g++ gcc && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/root/miniconda3/bin:$PATH

# init conda and update
RUN echo "Running $(conda --version)" && \
    conda init bash && . /root/.bashrc && \
    conda update conda

# set up conda environment
RUN conda create -n strda python=3.8
RUN echo "source activate strda" > ~/.bashrc
ENV PATH /root/miniconda3/envs/strda/bin:$PATH

# install dependencies
RUN pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install opencv-python==4.4.0.46 Pillow==7.2.0 opencv-python-headless==4.5.1.48 lmdb tqdm nltk six pyyaml

RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6

# get repository
WORKDIR /home
