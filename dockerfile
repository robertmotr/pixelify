FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /usr/src/app

COPY . .

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
RUN git submodule update --init --recursive \
    && git submodule update --remote --merge \
    && cd /app/external/googletest \
    && cmake . \
    && make -j$(nproc) \
    && make install \ 
    && cd /usr/src/app

# build pixelify
RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

#ENTRYPOINT [ "./build/pixelify" ]