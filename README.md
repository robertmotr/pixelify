# pixelify
Image processor that applies filters to images using CUDA. 

# TODO:
- basically just rewrite kernel from scratch, not sustainable
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
- add rotate option
- add flip option
- add zoom in/zoom out tool

# in progress:
- rewrite kernel to be speedy using shared memory + double parallelizing filter/image iteration
- rewrite kernel to comply to real world kernel conventions and use floats

# done:
- imgui implementation 
- test cases for all cases 