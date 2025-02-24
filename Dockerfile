ARG BASE_IMAGE=python:3.11-slim
FROM $BASE_IMAGE

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get -y install zsh git curl htop && \
    apt-get -y install gcc mono-mcs libglu1-mesa-dev libglib2.0-dev libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

# install project requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -U pip
RUN pip3 install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -f -g ${USER_GID} my_group && \
useradd -m -d /home/user -s /bin/zsh -g ${USER_GID} -u ${USER_UID} user

WORKDIR /workspaces
USER user
