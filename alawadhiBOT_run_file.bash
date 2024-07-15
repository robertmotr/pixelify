rm -r build
mkdir build
cd build

# Set environment variables for GCC 11
export CUDAHOSTCXX=/usr/bin/g++-11
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++

# Run CMake and Make
cmake -DCMAKE_BUILD_TYPE=Debug ..

make -j$(nproc)
