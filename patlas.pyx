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
    #define STB_DXT_IMPLEMENTATION
    #define QOI_IMPLEMENTATION
    #define QOI_NO_STDIO
    #ifndef _OPENMP
        #define omp_get_max_threads() 1
    #endif
    """

cdef extern from 'stb/stb_image.h' nogil:
    int stbi_info(const char* filename, int *x, int *y, int *comp)
    unsigned char* stbi_load(const char* filename, int *x, int *y, int *channels_in_file, int desired_channels)
    void stbi_set_flip_vertically_on_load(int flag_true_if_should_flip)
    const char* stbi_failure_reason()

cdef extern from 'stb/stb_rect_pack.h' nogil:
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

cdef extern from 'stb/stb_dxt.h' nogil:
    void stb_compress_dxt_block(unsigned char* dest, const unsigned char* src_rgba_four_bytes_per_pixel, int alpha, int mode)

cdef extern from 'qoi/qoi.h' nogil:
    ctypedef struct qoi_desc:
        unsigned int width
        unsigned int height
        unsigned char channels
        unsigned char colorspace
    
    void* qoi_encode(const void *data, const qoi_desc *desc, int *out_len)
    void* qoi_decode(const void *data, int size, qoi_desc *desc, int channels)

cpdef enum Heuristic:
    BL_SORTHEIGHT = 0
    BF_SORTHEIGHT

cpdef enum TextureFormat:
    RGBA8 = 0x1908
    DXT5 = 0x83F3

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
    cdef readonly TextureFormat texture_format 
    cdef int _dirty
    # 
    cdef stbrp_node* nodes
    cdef unsigned char* _atlas
    cdef array _cyatlas

    cdef unsigned char* _dxt5
    cdef array _cydxt5

    def __init__(self, side: int, pad: int=2, 
                 heuristic: Heuristic=Heuristic.BL_SORTHEIGHT,
                 texture_format: TextureFormat=TextureFormat.RGBA8):
        self.width = side
        self.height = side
        if texture_format == TextureFormat.DXT5 and (self.width % 4 != 0 or self.height % 4 != 0):
            raise RuntimeError('Each side of DXT5 texture must be a multiple of 4.')
        
        self.pad = pad
        self.num_nodes = 2 * side
        self.heuristic = heuristic
        self.texture_format = texture_format
        self.metadata = {'texture_format': texture_format, 'images': {}}
        self._dirty = 0

        self.nodes = <stbrp_node*> PyMem_RawMalloc(self.num_nodes * sizeof(stbrp_node))
        # we only call init once, so that we can re-use with another call to pack
        if self.nodes == NULL:
            raise RuntimeError('Unable to allocate stbrp_node memory.')
        stbrp_init_target(&self.context, self.width, self.height, self.nodes, self.num_nodes)
        stbrp_setup_heuristic(&self.context, self.heuristic)
        self._atlas = <unsigned char*> PyMem_RawCalloc(self.width * self.height * 4, sizeof(char))
        self._cyatlas = array((self.width, self.height, 4), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
        self._cyatlas.data = <char*> self._atlas

        if texture_format == TextureFormat.DXT5:
            self._dxt5 = <unsigned char*> PyMem_RawCalloc(self.width * self.height, sizeof(char))
            self._cydxt5 = array((self.width, self.height, 1), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
            self._cydxt5.data = <char*> self._dxt5
        
        stbi_set_flip_vertically_on_load(1) # set bottom-left as start


    cpdef pack(self, images: list[str]):
        # take list of image paths
        # return nothing (or just warning/err)
        cdef stbrp_rect* rects
        cdef int x, y, yy, channels_in_file, size, i, w, h
        cdef int n_images = len(images)
        cdef int max_threads = openmp.omp_get_max_threads()
        cdef int* xys # interleaved x,y (so e.g. xy[0] is x and xy[1] is y)
        cdef int thread_id, thread_idx # local ID & index

        # step 1: read image attributes
        cdef const char **im_names = to_cstring_array(images)
        cdef unsigned char* data
        cdef unsigned char* source_row
        cdef unsigned char* target_row
        try:
            rects = <stbrp_rect*> malloc(n_images * sizeof(stbrp_rect))
            xys = <int*> malloc(max_threads * 2 * sizeof(int))
            for i in range(n_images):
                if not stbi_info(im_names[i], &x, &y, &channels_in_file):
                    raise RuntimeError('Image property query failed. %s' % stbi_failure_reason())
                rects[i].id = i # unused
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
                thread_idx = thread_id * 2
                for i in prange(n_images, schedule='guided'):
                    data = stbi_load(im_names[i], &xys[thread_idx], &xys[thread_idx+1], &channels_in_file, 4) # force RGBA
                    if data is NULL:
                        with gil:
                            raise RuntimeError('Image failed to load. %s' % stbi_failure_reason())
                    
                    # conceptually from https://stackoverflow.com/a/12273365/2690232
                    # loop through source image rows
                    for yy in range(xys[thread_idx+1]):
                        source_row = &data[yy * xys[thread_idx] * 4]
                        # get the subset of the atlas we're writing this row to-- need to account for padding
                        # and global offset within atlas
                        # TODO: this doesn't work for non-square target images, but I can't reason through why?
                        target_row = &self._atlas[(rects[i].y + yy + self.pad) * self.width * 4 + (rects[i].x + self.pad) * 4]
                        memcpy(target_row, source_row, xys[thread_idx] * 4 * sizeof(char))

                    free(data) # done with the image now (TODO: should use STBI_FREE)
                
            # step 4: build up dict with keys
            for i in range(n_images):
                x = rects[i].x + self.pad
                y = rects[i].y + self.pad
                w = rects[i].w - 2*self.pad
                h = rects[i].h - 2*self.pad
                # TODO: store as array [u0 v0 u1 v1] or like this?
                # This is more descriptive, but more typing on the user side
                self.metadata['images'][op.splitext(op.basename(images[i]))[0]] = {'u0': x / <double>self.width,
                                                                                   'v0': y / <double>self.height,
                                                                                   'u1': (x + w) / <double>self.width,
                                                                                   'v1': (y + h) / <double>self.height}
            self._dirty = 1
        # all done (and/or failed), free
        finally:
            free(rects)
            free(xys)
            PyMem_RawFree(im_names)

    @property
    def atlas(self):
        if self.texture_format == TextureFormat.RGBA8:
            return self._cyatlas.memview
        elif self.texture_format == TextureFormat.DXT5:
            # we can't compress during atlas packing, because we don't know what adjacent pixels will be
            if self._dirty:
                self.compress_dxt5()
                self._dirty = 0
            
            return self._cydxt5.memview
    
    cdef void compress_dxt5(self) nogil:
        # see e.g. stbgl__compress https://github.com/nothings/stb/blob/c9064e317699d2e495f36ba4f9ac037e88ee371a/tests/caveview/stb_gl.h#L527
        cdef int i, j, y, y2, x
        #cdef unsigned char block[16*4]
        cdef unsigned char* block
        cdef int thread_id, thread_idx
        cdef int max_threads = openmp.omp_get_max_threads()

        block = <unsigned char*> malloc(16*4*max_threads)
        with parallel():
            thread_id = threadid()
            thread_idx = thread_id * 64
            for j in prange(0, self.width, 4, schedule='static'):
                x = 4
                for i in range(0, self.height, 4):
                    if (i + 3) >= self.width:
                        x = self.width - i
                    for y in range(4):
                        if (j + y >= self.height): break
                        memcpy(&block[thread_idx] + y*16, self._atlas + self.width*4*(j+y) + i*4, x*4)
                    if (x < 4):
                        if (x == 1):
                            for y2 in range(y):
                                memcpy(&block[thread_idx] + y2*16 + 4, &block[thread_idx] + y2*16, 4)
                                memcpy(&block[thread_idx] + y2*16 + 8, &block[thread_idx] + y2*16, 8)
                        elif (x == 2):
                            for y2 in range(y):
                                memcpy(&block[thread_idx] + y2*16 + 8, &block[thread_idx] + y2*16, 8)
                        elif (x == 3):
                            for y2 in range(y):
                                memcpy(&block[thread_idx] + y2*16 + 12, &block[thread_idx] + y2*16 + 4, 4)
                        else:
                            with gil:
                                raise RuntimeError('Something went wrong during compression.')

                    y2 = 0
                    for y in range(y, 4):
                        memcpy(&block[thread_idx] + y*16, &block[thread_idx] + y2*16, 16)
                        y2 = y2 + 1
                    
                    stb_compress_dxt_block(&self._dxt5[(j * self.width) + i*4], &block[thread_idx], 1, 10)
        free(block)

    def save(self, name: str):
        # TODO: option to use .png instead? Slower but smaller
        # dump the dictionary and atlas into a pickle
        # the atlas is encoded as .qoi
        cdef void* encoded
        cdef qoi_desc desc
        cdef int size
        cdef array temp

        desc.width = self.width
        desc.height = self.height
        desc.channels = 4
        desc.colorspace = 0 # unused

        encoded = qoi_encode(self._atlas, &desc, &size)
        if encoded is NULL:
            raise RuntimeError('Failed to encode atlas.')
        
        try:
            temp = array((size,), itemsize=sizeof(char), format='b', allocate_buffer=False)
            temp.data = <char *> encoded
            with open(f'{name}.patlas', 'wb') as f:
                pkl.dump((zlib.compress(memoryview(temp), 1), self.metadata), f, 4) # TODO: pkl.DEFAULT_PROTOCOL?
        finally:
            free(encoded) # TODO: should be QOI_FREE

    def __dealloc__(self):
        PyMem_RawFree(self.nodes)
        PyMem_RawFree(self._atlas)
        PyMem_RawFree(self._dxt5) # unconditional?


cpdef load(filename: str):
    # load a .patlas file
    cdef bytes raw_atlas
    cdef dict metadata
    cdef int texture_format
    with open(filename, 'rb') as f:
        raw_atlas, metadata = pkl.load(f)

    cdef const unsigned char[:] mview = zlib.decompress(raw_atlas)
    cdef int len_data = mview.shape[0]
    cdef qoi_desc desc
    cdef void* temp = qoi_decode(<const void*> &mview[0], len_data, &desc, 4)
    if temp is NULL:
        raise RuntimeError('Unable to load .qoi image from .patlas file.')
    
    cdef array _atlas = array((desc.width, desc.height, 4), mode='c', itemsize=sizeof(char), format='B', allocate_buffer=False)
    _atlas.data = <char *> temp
    _atlas.callback_free_data = free
    return _atlas, metadata
