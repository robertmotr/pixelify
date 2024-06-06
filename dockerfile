FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /usr/src/app

COPY . .

# in order to configure the nvidia-container-toolkit repository, we need to install some dependencies
RUN apt-get update && \
    apt-get install -y \ 
        ca-certificates \
        gnupg \
        lsb-release \
        curl \
        software-properties-common \
        && rm -rf /var/lib/apt/lists/*

# configure production repository for nvidia-container-toolkit
RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
  sed -i -e '/experimental/ s/^#//g' /etc/apt/sources.list.d/nvidia-container-toolkit.list 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        valgrind \
        nvidia-container-toolkit \
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
    && cd external/googletest \
    && cmake . \
    && make -j$(nproc) \
    && make install \ 
    && cd /usr/src/app

# build pixelify
RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

#ENTRYPOINT [ "./build/pixelify" ]