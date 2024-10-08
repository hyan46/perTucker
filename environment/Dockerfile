# hash:sha256:918b6b6d9d7058eab92e5cbfa3aec1d3dbb0aae58881ee483ee06854aedd13b4
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        matplotlib==3.9.2 \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scipy==1.12.0 \
        seaborn==0.13.2 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    tensorly==0.8.1
