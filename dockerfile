FROM rocker/cuda

RUN apt-get update && \
    apt-get install -y \
        cmake \
        cuda \
        xserver-xorg \ 
        ubuntu-drivers-common \ 
        apt-utils \
        git \
        build-essential \
        libglfw3-dev \
        libglew-dev \
        libexiv2-dev \
        libgtest-dev \
        && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -y && apt-get upgrade -y 

COPY . /app
WORKDIR /app

RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

CMD ["./build/pixelify"]
