# use basic conda image
FROM continuumio/miniconda3

# install nginx and others for hosting
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    ca-certificates \
    wget \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# set workdir
WORKDIR /app

# copy main content to image
COPY libcom /app/libcom

# install conda env Libcom
WORKDIR /app/libcom/requirements
RUN conda env create -f libcom.yaml

# activate Libcom and continue
SHELL ["conda", "run", "-n", "Libcom", "/bin/bash", "-c"]

RUN apt-get update && apt-get install -y g++

RUN apt-get update && apt-get install -y ninja-build

# install dependencies 
RUN pip install -r runtime.txt
# -i https://pypi.tuna.tsinghua.edu.cn/simple

# install taming-transformers
WORKDIR /app/libcom/libcom/controllable_composition/source/ControlCom/src/taming-transformers
RUN python setup.py install

# install trilinear and use copy complied egg file into site-package for facilities
WORKDIR /app/libcom/libcom/image_harmonization/source/trilinear_cpp
RUN python setup.py install
RUN rm -rf /opt/conda/envs/Libcom/lib/python3.8/site-packages/trilinear-0.0.0-py3.8-linux-x86_64.egg
COPY libcom/requirements/trilinear-0.0.0-py3.8-linux-x86_64.egg /opt/conda/envs/Libcom/lib/python3.8/site-packages/

# set container work dir
WORKDIR /app/libcom/libcom

# install gunicorn
RUN pip --no-cache-dir install \
uvicorn \
fastapi \
gunicorn \
boto3 \
flask \
python-multipart

# install typing-extensions
RUN pip install --upgrade typing-extensions

RUN apt-get update && apt-get install -y vim


# Set some environment variables. PYTHONUNBUFFERED keeps Python from buffering our standard
# output stream, which means that logs can be delivered to the user quickly. PYTHONDONTWRITEBYTECODE
# keeps Python from writing the .pyc files which are unnecessary in this case. We also update
# PATH so that the train and serve programs are found when the container is invoked.

ENV LANG C.UTF-8

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH /usr/local/bin:$PATH
ENV PATH="/app/libcom/libcom:${PATH}"
ENV PYTHONPATH=/app/libcom

RUN echo "conda activate Libcom" >>~/.bashrc

# set ENTRYPOINT to run serve in env Libcom
ENTRYPOINT ["/bin/bash", "-c", "source activate Libcom && exec \"$@\"", "--"]
