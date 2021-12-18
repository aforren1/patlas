from patlas import AtlasPacker
from timeit import default_timer
import matplotlib.pyplot as plt

x = AtlasPacker(2048, pad = 1)

t0 = default_timer()
for i in range(5):
    x.pack(['images/alex.png', 'images/kazoo.jpg'])

# funnily enough, packing all at once does worse than incremental?
#x.pack(['images/alex.png', 'images/kazoo.jpg']*4)
t1 = default_timer() - t0
print(f'Packing time: {t1} sec')
y = x.atlas

print(y.shape)
print(x.locations)
plt.imshow(y, origin='lower')
plt.show()
