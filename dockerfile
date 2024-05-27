FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        xauth \ 
        xorg \
        xserver-xorg \
        cmake \
        git \
        wget \
        unzip \
        pkg-config \
        libx11-dev \
        libxi-dev \
        libgl1-mesa-dev \
        libglu1-mesa-dev \
        libglfw3-dev \
        libglew-dev \
        libexiv2-dev \
        && rm -rf /var/lib/apt/lists/*

# install googletest
RUN git clone https://github.com/google/googletest.git /usr/src/googletest \
    && cd /usr/src/googletest \
    && cmake . \
    && make -j$(nproc) \
    && make install

WORKDIR /usr/src/app

COPY . .

# build pixelify
RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

#ENTRYPOINT [ "./build/pixelify" ]