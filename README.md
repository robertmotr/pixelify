# Pixelify
### Image processor that applies filters to images using CUDA. 

# Installation steps:
## On Linux:

### Required dependencies:
- git 
- GoogleTest
- OpenGL
- GLEW
- glfw3
- exiv2
- gcc
- CUDA

### Optional dependencies:
- docker
- nvidia-container-toolkit

Install all these dependencies using your package manager, depending on which distro you have.

`git clone` the repository:
```
git clone https://github.com/robertmotr/pixelify.git
```

Inside your locally cloned repository, update all the external libraries:
```
git submodule update --init --recursive --remote
```
### Using Docker:
If you decide to use docker (assuming you installed the optional dependencies), then start a docker daemon using `dockerd`, and then run the following build scripts:
```
sh base-build.sh && sh full-build.sh
```
You may need to grant the Docker container access to your host's X server. In that case, you need to run the following command: `xhost +local:docker`

Finally run `sh run-dockerized.sh` to launch Pixelify through Docker. If you decide not to use Docker, then see the next section.

### Building and running natively:
Simply run the following command:
 ```mkdir build && cd build && cmake -{nproc} -DCBUILD_TYPE=Release ..```
Once the build finishes, launch the application by executing `./pixelify`.

## On Windows:
The latest executables for Windows are released under 
`https://github.com/robertmotr/pixelify/releases/`. (TODO, this has not been done yet).

# TODO:
- add analytics tab and keep track of rendering times, maybe have a naive CPU comparison too
- keyboard shortcuts for tabbing between original/preview
- parse filter file -> get filter array of pixels 
- support anti-aliasing methods + other applications aside from filters?
- cuda <--> amd, work on any GPU, maybe use HIP or some transpiling method

# In progress:
- redo all tests to fix bugs and have safety
- realtime rendering if fast enough (maybe?)

# Done:
- add zoom in/zoom out/pan tool + texture map
- profile texture/shared/constant memory for filter and use whichever one is faster
- move allocating image data and stuff to opening input file to make rendering seem faster,
- speed up kernel by using cached constant memory for both the filter_dimension and filter_data
- get rid of device_filter allocations because its useless now with constant memory
- get other_kernel functionality working again
- fix segfault from kernel formulas
- fix kernel formulas not producing results that "look right"
- add fits/xmpp data to image description
- add multiple filter passes option
- add pixel inspector
- add filter size slider, i.e 2x2 -> 9x9
- set filter size slider to be dependent on filter selected, i.e identity can only have 3x3 
- optimize kernels to get ~200ms runtime preferably even lower (update: getting consistently sub 20ms which is fantastic)
- imgui implementation 