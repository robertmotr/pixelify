FROM pixelify-base

WORKDIR /usr/src/app

COPY . .

# build pixelify
RUN rm -rf build && mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    make -j$(nproc)

#ENTRYPOINT [ "./build/pixelify" ]