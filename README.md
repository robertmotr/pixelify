# pixelify
Image processor that applies filters to images using CUDA. 

# TODO:
- add multiple filter passes option
- add pixel inspector
- keyboard shortcuts for tabbing between original/preview
- parse filter file -> get filter array of pixels 
- support anti-aliasing methods + other applications aside from filters?
- cuda <--> amd, work on any GPU, maybe use HIP or some transpiling method
- add contrast, should be easy with CUB histogram
- add fits/xmpp data to image description, also add file modification date, etc
- add gamma option
- add zoom in/zoom out tool
- add filter size slider, i.e 2x2 -> 9x9
- use graphics interoperability, i.e make cuda register from imgui texture instead of allocating again
- profile texture/shared memory for filter and use whichever one is faster

# in progress:

- realtime rendering if fast enough (maybe?)
- add rotate kernel
- add shear option + kernel
- add flip option
- set filter size slider to be dependent on filter selected, i.e identity can only have 3x3 

# done:
- optimize kernels to get ~200ms runtime preferably even lower (update: getting consistently sub 50ms which is fantastic)
- make pixels use texture memory for faster reads
- make texture memory use shorts because it supports only <= x bits
- imgui implementation 
- test cases for all cases 
- rewrite filter to use shared memory
