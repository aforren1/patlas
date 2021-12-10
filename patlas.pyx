#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True
cimport cython
from cpython.mem cimport PyMem_Malloc, PyMem_Free, PyMem_Calloc, PyMem_Realloc
import os.path as op

cdef extern from *:
    """
    #define STB_RECT_PACK_IMPLEMENTATION
    #define STB_IMAGE_IMPLEMENTATION
    #define STBI_MALLOC PyMem_Malloc
    #define STBI_FREE PyMem_Free
    #define STBI_REALLOC PyMem_Realloc
    """

cdef extern from '../_stb/stb_image.h':
    int stbi_info(const char* filename, int *x, int *y, int *comp)
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    const char* stbi_failure_reason()

cdef extern from "../_stb/stb_rect_pack.h":
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
cdef class AtlasPacker:
    cdef stbrp_context* context
    cdef int width
    cdef int height
    cdef int pad
    cdef int num_nodes
    cdef int heuristic
    # 
    cdef int num_keys
    cdef stbrp_node* nodes
    cdef unsigned char* atlas

    def __init__(self, width, height, pad=2, heuristic=Heuristic.DEFAULT):
        self.width = width
        self.height = height
        self.pad = pad
        self.num_nodes = width
        self.heuristic = heuristic
        self.keys = []
        self.num_keys = 0

    def pack(self, images):
        # take list of image paths
        # return nothing for now (or just warning/err)-- on request, give memoryview & dict
        if self.context == NULL:
            self.nodes = <stbrp_node*> PyMem_Malloc(self.num_nodes * sizeof(stbrp_node))
            # we only call init once, so that we can re-use with another call to pack
            stbrp_init_target(self.context, self.width, self.height, self.nodes, self.num_nodes)
            stbrp_setup_heuristic(self.context, self.heuristic)
            self.atlas = <unsigned char*> PyMem_Calloc(self.width * self.height * 4, sizeof(char))


        # step 1: read image attributes
        potential_keys = []
        cdef int counter = 0
        cdef stbrp_rect* rects
        cdef int x, y, channels_in_file, size
        try:
            rects = <stbrp_rect*> PyMem_Malloc(len(images) * sizeof(stbrp_rect))
            for im in images:
                if not stbi_info(im, &x, &y, &channels_in_file):
                    raise RuntimeError('Image property query failed. %s' % stbi_failure_reason())
                potential_keys.append(op.splitext(op.basename(im))[0])
                rects[counter].id = counter
                rects[counter].w = x + self.pad
                rects[counter].h = y + self.pad
                counter += 1

            # step 2: pack the rects
            if not stbrp_pack_rects(self.context, rects, counter):
                raise RuntimeError('Failed to pack rectangles.')

            # step 3: read in images and stick in memoryview, accounting for padding 
            # see https://stackoverflow.com/q/12273047/2690232
            # for padding ideas
            
            for im in images:
                cdef unsigned char* data = stbi_load(filename, &x, &y, &channels_in_file, 4) # force RGBA
                if data is NULL:
                    raise RuntimeError('Memory failed to load. %s' % stbi_failure_reason())
                

        # all done (and/or failed), free rects
        finally:
            PyMem_Free(rects)


    def __dealloc__(self):
        PyMem_Free(self.nodes)
        PyMem_Free(self.atlas)
