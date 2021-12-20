from patlas import AtlasPacker
from timeit import default_timer
import pathlib

# stress
N = 100; dim = 2 ** 13
# non-stress
#N = 4; dim = 2 ** 11

ims = [str(pathlib.Path(__file__).parent.resolve() / 'images' / x) for x in ['alex.png', 'kazoo.jpg']]
x = AtlasPacker(dim, pad = 1)

t0 = default_timer()
for i in range(N):
    x.pack(ims)

t1 = default_timer() - t0
print(f'Packing time: {t1} sec')

# Pack all at once (with OpenMP, should be much faster)
z = AtlasPacker(dim, pad = 1)
t0 = default_timer()
z.pack(ims*N)
t1 = default_timer() - t0
print(f'Packing time: {t1} sec')

if False:
    import matplotlib.pyplot as plt
    plt.imshow(x.atlas, origin='lower')
    plt.show()

    plt.imshow(z.atlas, origin='lower')
    plt.show()
