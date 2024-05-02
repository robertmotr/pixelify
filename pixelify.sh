#!/bin/bash
# no linux/unix system? yikes :( maybe you should hop off windows loser

git submodule update --init --recursive
cd build && ninja clean && ninja -j8

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi
elif [ $? -eq 0 ]; then
    echo "Build successful"

    if [ $1 == "tests" ]; then
        ./tests
        if [ $? -ne 0 ]; then
            echo "Tests failed"
            exit 1
        fi
        else 
            echo "Tests successful"
    fi
    else
        ./pixelify
fi