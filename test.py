from contextlib import ContextDecorator
from patlas import AtlasPacker, load, TextureFormat
from timeit import default_timer
import pathlib

# stress
N = 100; dim = 2 ** 13
# non-stress
#N = 4; dim = 2 ** 11
try:
    pth = __file__
except NameError:
    pth = '.'

class timer(ContextDecorator):
    def __init__(self, strval):
        self.sv = strval
    def __enter__(self):
        self.t0 = default_timer()
        return self
    def __exit__(self, *exc):
        t1 = default_timer() - self.t0
        print(f'{self.sv} took {t1} seconds.')

pth = pathlib.Path(pth).parent.resolve()
ims = [str(pth / 'images' / x) for x in ['alex.png', 'kazoo.jpg']]
x = AtlasPacker(dim, pad = 1)

with timer('Multi'):
    for i in range(N):
        x.pack(ims)

# Pack all at once (with OpenMP, should be much faster)
z = AtlasPacker(dim, pad = 1)
with timer('Single'):
    z.pack(ims*N)

with timer('Pickle'):
    x.save(str(pth / 'foo'))

with timer('Load'):
    loaded_atlas, loaded_locs = load(str(pth / 'foo.patlas'))

assert bytes(loaded_atlas) == bytes(x.atlas)

# DXT5
w = AtlasPacker(dim, pad=1, texture_format=TextureFormat.DXT5)
with timer('DXT5'):
    w.pack(ims*N)

with timer('DXT5 retrieval'):
    tmp = w.atlas

with timer('DXT5 retrieval (cached)'):
    tmp = w.atlas


if True:
    import matplotlib.pyplot as plt
    from PIL import Image
    import numpy as np
    plt.imshow(x.atlas, origin='lower')
    plt.show()

    plt.imshow(z.atlas, origin='lower')
    plt.show()

    plt.imshow(w.atlas, origin='lower')
    plt.show()

    im = Image.fromarray(np.array(x.atlas))
    
    with timer('PIL save PNG'):
        im.save('test.png')

    with timer('PIL load PNG'):
        with Image.open('test.png') as im:
            a = np.asarray(im)
