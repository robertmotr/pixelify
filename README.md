# pixelify
Image processor that applies filters to images using CUDA. 

# TODO:
- add analytics tab and keep track of rendering times, maybe have a naive CPU comparison too
- keyboard shortcuts for tabbing between original/preview
- parse filter file -> get filter array of pixels 
- support anti-aliasing methods + other applications aside from filters?
- cuda <--> amd, work on any GPU, maybe use HIP or some transpiling method
- add contrast, should be easy with CUB histogram
- add gamma option
- add flip option
- dynamically determine an optimized grid size based on GPU + image

# in progress:
- redo all tests to fix bugs and have safety
- stop using CUB for image reductions and just write kernel myself
- add invert image, threshold
- realtime rendering if fast enough (maybe?)
- add rotate kernel
- add shear option + kernel

# done:
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


# Building:
## Linux:

## Required dependencies:
- git 
- GoogleTest
- https://wiki.archlinux.org/title/OpenGL
- GLEW
- glfw3
- exiv2
- gcc
- CUDA

Install all these dependencies using your package manager, depending on which distro you have.

Git clone the repository:
```
git clone https://github.com/robertmotr/pixelify.git
```

Install the following packages using your AUR helper (on Arch) or your package manager on other distros:
```
cuda glfw glew exiv2
```
Then, run the "setup.bash" file in the home directory.
