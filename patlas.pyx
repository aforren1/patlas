#cython: c_string_type=unicode, c_string_encoding=ascii
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: infertypes=True
#cython: initializedcheck=False
#cython: cdivision=True
cimport cython
cimport openmp
from cython.view cimport array
from cpython.mem cimport PyMem_RawMalloc, PyMem_RawFree, PyMem_RawRealloc
from libc.stdlib cimport malloc, free
from cython.parallel import prange, parallel, threadid
from libc.string cimport memcpy

import pickle as pkl
import os.path as op
import zlib

cdef extern from "Python.h":
    char* PyUnicode_AsUTF8(object unicode)
    # wasn't fixed until a week ago, so not in any release yet...
    void* PyMem_RawCalloc(size_t nelem, size_t elsize)

# TODO future: custom malloc (e.g. malloc once based off max image size, then keep pulling from that until done?)
# but I suppose that doesn't parallelize well...
cdef extern from *:
    """
    #define STB_RECT_PACK_IMPLEMENTATION
    #define STB_IMAGE_IMPLEMENTATION
    #define QOI_IMPLEMENTATION
    #ifndef _OPENMP
        #define omp_get_max_threads() 1
    #endif
    //#define STBI_MALLOC PyMem_RawMalloc
    //#define STBI_FREE PyMem_RawFree
    //#define STBI_REALLOC PyMem_RawRealloc
    """

cdef extern from 'stb/stb_image.h' nogil:
    int stbi_info(const char* filename, int *x, int *y, int *comp)
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    void stbi_set_flip_vertically_on_load(int flag_true_if_should_flip)
    const char* stbi_failure_reason()

cdef extern from "stb/stb_rect_pack.h" nogil:
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

cdef extern from 'qoi/qoi.h':
    ctypedef struct qoi_desc:
        unsigned int width
        unsigned int height
        unsigned char channels
        unsigned char colorspace
    
    void* qoi_encode(const void *data, const qoi_desc *desc, int *out_len)
    void* qoi_decode(const void *data, int size, qoi_desc *desc, int channels)

cpdef enum Heuristic:
    DEFAULT = 0
    BL_SORTHEIGHT = DEFAULT
    BF_SORTHEIGHT

# from https://stackoverflow.com/a/54081075/2690232
cdef char ** to_cstring_array(list_str):
    cdef int len_list = len(list_str)
    cdef int i
    cdef char **ret = <char **>PyMem_RawMalloc(len_list * sizeof(char *))
    for i in range(len_list):
        ret[i] = PyUnicode_AsUTF8(list_str[i])
    return ret

@cython.no_gc_clear
@cython.final # allow nogil for pack
cdef class AtlasPacker:
    cdef stbrp_context context
    cdef int width
    cdef int height
    cdef int pad
    cdef int num_nodes
    cdef int heuristic
    cdef readonly dict metadata
    # 
    cdef stbrp_node* nodes
    cdef unsigned char* _atlas

    def __init__(self, side: int, pad: int=2, heuristic: Heuristic=Heuristic.DEFAULT):
        self.width = side
        self.height = side
        self.pad = pad
        self.num_nodes = 2 * side
        self.heuristic = heuristic
        self.metadata = {}

        self.nodes = <stbrp_node*> PyMem_RawMalloc(self.num_nodes * sizeof(stbrp_node))
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
        cdef stbrp_rect* rects
        cdef int x, y, yy, channels_in_file, size, _id, i
        cdef int n_images = len(images)
        cdef int max_threads = openmp.omp_get_max_threads()
        cdef int* xs
        cdef int* ys
        cdef int thread_id # local ID

        # step 1: read image attributes
        cdef const char **im_names = to_cstring_array(images)
        cdef unsigned char* data
        cdef unsigned char* source_row
        cdef unsigned char* target_row
        try:
            rects = <stbrp_rect*> malloc(n_images * sizeof(stbrp_rect))
            xs = <int*> malloc(max_threads * sizeof(int))
            ys = <int*> malloc(max_threads * sizeof(int))
            for i in range(n_images):
                if not stbi_info(im_names[i], &x, &y, &channels_in_file):
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
            with nogil, parallel():
                thread_id = threadid()
                for i in prange(n_images, schedule='guided'):
                    _id = rects[i].id
                    data = stbi_load(im_names[_id], &xs[thread_id], &ys[thread_id], &channels_in_file, 4) # force RGBA
                    if data is NULL:
                        with gil:
                            raise RuntimeError('Memory failed to load. %s' % stbi_failure_reason())
                    
                    # conceptually from https://stackoverflow.com/a/12273365/2690232
                    # loop through source image rows
                    for yy in range(ys[thread_id]):
                        source_row = &data[yy * xs[thread_id] * 4]
                        # get the subset of the atlas we're writing this row to-- need to account for padding
                        # and global offset within atlas
                        # TODO: this doesn't work for non-square target images, but I can't reason through why?
                        target_row = &self._atlas[(rects[_id].y + yy + self.pad) * self.width * 4 + (rects[_id].x + self.pad) * 4]
                        memcpy(target_row, source_row, xs[thread_id] * 4 * sizeof(char))
                    
                    free(data) # done with the image now (TODO: should use STBI_FREE)
                
            # step 4: build up dict with keys
            for i in range(n_images):
                _id = rects[i].id
                # TODO: should this be uv coords instead?
                self.metadata[op.splitext(op.basename(images[_id]))[0]] = {'x': rects[_id].x + self.pad,
                                                                           'y': rects[_id].y + self.pad,
                                                                           'w': rects[_id].w - 2*self.pad,
                                                                           'h': rects[_id].h - 2*self.pad}

        # all done (and/or failed), free
        finally:
            free(rects)
            free(xs)
            free(ys)
            PyMem_RawFree(im_names)


    @property
    def atlas(self):
        cdef array _atlas = array((self.width, self.height, 4), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
        _atlas.data = <char*> self._atlas
        # we manage the memory internally, so no need to set free callback?
        return _atlas.memview

    def save(self, name: str):
        # TODO: option to use .png instead? Slower but smaller
        # dump the dictionary and atlas into a pickle
        # the atlas is encoded as .qoi
        cdef void* encoded
        cdef qoi_desc desc
        cdef int size

        desc.width = self.width
        desc.height = self.height
        desc.channels = 4
        desc.colorspace = 0 # unused

        encoded = qoi_encode(self._atlas, &desc, &size)
        if encoded is NULL:
            raise RuntimeError('Failed to encode atlas.')
        
        cdef array temp = array((size,), itemsize=sizeof(char), format='b', allocate_buffer=False)
        temp.data = <char *> encoded
        with open(f'{name}.patlas', 'wb') as f:
            pkl.dump((zlib.compress(memoryview(temp), 1), self.metadata), f, 4) # TODO: pkl.DEFAULT_PROTOCOL?
        
        free(encoded) # TODO: should be QOI_FREE

    def __dealloc__(self):
        PyMem_RawFree(self.nodes)
        PyMem_RawFree(self._atlas)


cpdef load(filename: str):
    # load a .patlas file
    with open(filename, 'rb') as f:
        raw_atlas, locations = pkl.load(f)
        raw_atlas = zlib.decompress(raw_atlas)
    
    cdef int len_data = len(raw_atlas)
    cdef const unsigned char[:] mview = raw_atlas
    cdef qoi_desc desc
    cdef void* temp = qoi_decode(<const void*> &mview[0], len_data, &desc, 4)
    if temp is NULL:
        raise RuntimeError('Unable to load .qoi image from .patlas file.')
    
    cdef array _atlas = array((desc.width, desc.height, 4), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
    _atlas.data = <char *> temp
    _atlas.callback_free_data = free
    return _atlas, locations
