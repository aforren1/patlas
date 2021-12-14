#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True
cimport cython
from cython.view cimport array
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree, PyMem_RawRealloc
#from cython.parallel import prange
from libc.string cimport memcpy
import os.path as op

cdef extern from "Python.h":
    # wasn't fixed until a week ago, so not in any release yet...
    void* PyMem_RawCalloc(size_t nelem, size_t elsize)

cdef extern from *:
    """
    #define STB_RECT_PACK_IMPLEMENTATION
    #define STB_IMAGE_IMPLEMENTATION
    #define STBI_MALLOC PyMem_RawMalloc
    #define STBI_FREE PyMem_RawFree
    #define STBI_REALLOC PyMem_RawRealloc
    """

cdef extern from 'stb/stb_image.h':
    int stbi_info(const char* filename, int *x, int *y, int *comp)
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    void stbi_set_flip_vertically_on_load(int flag_true_if_should_flip)
    const char* stbi_failure_reason()

cdef extern from "stb/stb_rect_pack.h":
    struct stbrp_context:
        pass

    struct stbrp_node:
        pass

    struct stbrp_rect:
        int id
        int w, h
        int x, y
        int was_packed

    int stbrp_pack_rects(stbrp_context *context, stbrp_rect *rects, int num_rects)
    void stbrp_init_target(stbrp_context *context, int width, int height, stbrp_node *nodes, int num_nodes)
    void stbrp_setup_heuristic(stbrp_context *context, int heuristic)

cpdef enum Heuristic:
    DEFAULT = 0
    BL_SORTHEIGHT = DEFAULT
    BF_SORTHEIGHT

@cython.no_gc_clear
@cython.final # allow nogil for pack
cdef class AtlasPacker:
    cdef stbrp_context context
    cdef int width
    cdef int height
    cdef int pad
    cdef int num_nodes
    cdef int heuristic
    cdef readonly dict locs
    # 
    cdef stbrp_node* nodes
    cdef unsigned char* _atlas

    def __init__(self, width: int, height: int, pad: int=2, heuristic: Heuristic=Heuristic.DEFAULT):
        self.width = width
        self.height = height
        self.pad = pad
        self.num_nodes = width
        self.heuristic = heuristic
        self.locs = {}

        self.nodes = <stbrp_node*> PyMem_RawMalloc(2 * self.num_nodes * sizeof(stbrp_node))
        # we only call init once, so that we can re-use with another call to pack
        if self.nodes == NULL:
            raise RuntimeError('Unable to allocate stbrp_node memory.')
        stbrp_init_target(&self.context, self.width, self.height, self.nodes, self.num_nodes)
        stbrp_setup_heuristic(&self.context, self.heuristic)
        self._atlas = <unsigned char*> PyMem_RawCalloc(self.width * self.height * 4, sizeof(char))
        stbi_set_flip_vertically_on_load(1) # set bottom-left as start


    cpdef pack(self, images: list[str]):
        # take list of image paths
        # return nothing for now (or just warning/err)-- on request, give memoryview & dict
        # TODO: release GIL
        cdef stbrp_rect* rects
        cdef int x, y, yy, channels_in_file, size, _id, n_images
        n_images = len(images)

        # step 1: read image attributes
        cdef unsigned char* data
        cdef unsigned char* source_row
        cdef unsigned char* target_row
        try:
            rects = <stbrp_rect*> PyMem_RawMalloc(len(images) * sizeof(stbrp_rect))
            for i in range(n_images):
                if not stbi_info(images[i], &x, &y, &channels_in_file):
                    raise RuntimeError('Image property query failed. %s' % stbi_failure_reason())
                rects[i].id = i
                rects[i].w = x + 2 * self.pad
                rects[i].h = y + 2 * self.pad

            # step 2: pack the rects
            if not stbrp_pack_rects(&self.context, rects, n_images):
                raise RuntimeError('Failed to pack rectangles. Try again with a larger atlas?')

            # step 3: read in images and stick in memoryview, accounting for padding 
            # see https://stackoverflow.com/q/12273047/2690232
            # for padding ideas
            # TODO: prange outer or inner loop (or both??)?
            for i in range(n_images):
                _id = rects[i].id
                data = stbi_load(images[_id], &x, &y, &channels_in_file, 4) # force RGBA
                if data is NULL:
                    raise RuntimeError('Memory failed to load. %s' % stbi_failure_reason())
                
                # conceptually from https://stackoverflow.com/a/12273365/2690232
                # loop through source image rows
                for yy in range(y):
                    source_row = &data[yy * x * 4]
                    # get the subset of the atlas we're writing this row to-- need to account for padding
                    # and global offset within atlas
                    target_row = &self._atlas[(yy + rects[_id].y + self.pad) * self.width * 4 + (rects[_id].x + self.pad) * 4]
                    memcpy(target_row, source_row, x * 4 * sizeof(char))
                
                PyMem_RawFree(data) # done with the image now (TODO: should use STBI_FREE)
            
            # step 4: build up dict with keys
            for i in range(n_images):
                _id = rects[i].id
                self.locs[op.splitext(op.basename(images[_id]))[0]] = {'x': rects[_id].x + self.pad, 
                                                                       'y': rects[_id].y + self.pad, 
                                                                       'w': rects[_id].w - 2*self.pad, 
                                                                       'h': rects[_id].h - 2*self.pad}

        # all done (and/or failed), free rects
        finally:
            PyMem_RawFree(rects)


    @property
    def atlas(self):
        cdef array _atlas = array((self.width, self.height, 4), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
        _atlas.data = <char*> self._atlas
        # we manage the memory internally, so no need to set free callback?
        return _atlas.memview
    
    @property
    def locations(self):
        return self.locs

    def __dealloc__(self):
        PyMem_RawFree(self.nodes)
        PyMem_RawFree(self._atlas)
