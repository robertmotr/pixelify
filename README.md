# Pixelify
### Image processor that applies filters to images using CUDA. 

# Installation steps:
## Linux:
- Ensure you have CUDA, gcc, CMake, OpenGL, glfw and GLEW installed.
- `git clone` the repository, then run the following inside of it:
```
mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release ../ && make -j${nproc}
```
- Finally, run the executable `pixelify` using `./pixelify` to start the app.

### Using Docker:
- Ensure that docker is installed, and that the repository has been cloned.
- Run the following scripts inside the repo: `sh full-build.sh && sh run-dockerized.sh`

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


# Installation:
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

If you decide to use docker (assuming you installed the optional dependencies), then start a docker daemon using `dockerd`, and then run the following build scripts:
```
sh base-build.sh && sh full-build.sh
```

Finally run `sh run-dockerized.sh` to launch Pixelify through Docker. If you decide not to use Docker, then run the following commands: 
`mkdir build && cd build && cmake -j${nproc} -DCBUILD_TYPE=Release ..`
Once the build finishes, launch the application by executing `./pixelify`.

## On Windows:
The latest executables for Windows are released under 
`https://github.com/robertmotr/pixelify/releases/`.