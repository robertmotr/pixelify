# pixelify
Image processor that applies filters to images using CUDA. 

# TODO:
- add multiple filter passes option
- add pixel inspector
- keyboard shortcuts for tabbing between original/preview
- realtime rendering if fast enough (maybe?)
- parse filter file -> get filter array of pixels 
- support anti-aliasing methods + other applications aside from filters?
- cuda <--> amd, work on any GPU, maybe use HIP or some transpiling method
- add contrast, should be easy with CUB histogram
- add fits/xmpp data to image description, also add file modification date, etc
- add gamma option
- add rotate kernel
- add flip option
- add shear option + kernel
- add zoom in/zoom out tool
- add filter size slider, i.e 2x2 -> 9x9
- set filter size slider to be dependent on filter selected, i.e identity can only have 3x3 

# in progress:
- rewrite kernel to be speedy using shared memory

# done:
- imgui implementation 
- test cases for all cases 