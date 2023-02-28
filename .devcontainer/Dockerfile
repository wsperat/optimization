FROM ubuntu:20.04

RUN apt update && \
    apt install -y build-essential wget git && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR /opt/conda

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH


COPY runtime_10-3.yaml .
RUN conda env create -n runtime_10-3 --file runtime_10-3.yaml
