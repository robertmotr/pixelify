FROM rocker/cuda

RUN apt-get update && \
    apt-get install -y \
        cuda \ 
        cmake \
        xserver-xorg \
        ubuntu-drivers-common \
        apt-utils \
        git \
        build-essential \
        libglfw3-dev \
        libglew-dev \
        libexiv2-dev \
        libgtest-dev \
        nvidia-container-toolkit \ 
        && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get upgrade -y 

COPY . /app
WORKDIR /app

RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

ARG uid=1000
RUN useradd -m -s /bin/bash -u $uid developer

RUN mkdir -p /etc/sudoers.d
RUN echo 'developer ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/developer

USER developer
ENV HOME /home/developer
ENV DISPLAY :1

CMD ["build/pixelify"]
