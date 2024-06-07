# Description: Build the base image for the pixelify project.
sudo docker build -f dockerfile.base -t pixelify-base .
echo "Base image built."
