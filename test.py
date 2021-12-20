from patlas import AtlasPacker
from timeit import default_timer

# stress
N = 400; dim = 2 ** 14
#N = 4; dim = 2 ** 11

x = AtlasPacker(dim, pad = 1)

t0 = default_timer()
for i in range(N):
    x.pack(['images/alex.png', 'images/kazoo.jpg'])

t1 = default_timer() - t0
print(f'Packing time: {t1} sec')
# funnily enough, packing all at once does worse than incremental?
z = AtlasPacker(dim, pad = 1)
t0 = default_timer()
z.pack(['images/alex.png', 'images/kazoo.jpg']*N)
t1 = default_timer() - t0
print(f'Packing time: {t1} sec')

if False:
    import matplotlib.pyplot as plt
    plt.imshow(x.atlas, origin='lower')
    plt.show()

    plt.imshow(z.atlas, origin='lower')
    plt.show()
