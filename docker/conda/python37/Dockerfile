FROM nvidia/cuda:10.1-cudnn7-devel
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    cmake \
    libblas3 \
    libblas-dev \
    libxext6 \
    libgl1-mesa-glx \
    libxrender-dev \
    && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc
RUN conda update -n base -c defaults conda
RUN conda create -n py37 python=3.7 conda && \
    . /opt/conda/etc/profile.d/conda.sh && \
    conda init bash && \
    conda activate py37 && \
    conda install -c rdkit rdkit && \
    pip install pytest mock

ADD conda-entrypoint.sh /conda-entrypoint.sh
ENTRYPOINT [ "/conda-entrypoint.sh" ]
