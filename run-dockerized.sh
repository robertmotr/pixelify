sudo docker run -e current_dir=/usr/src/app --runtime=nvidia --gpus all -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix/ -it pixelify 