from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

im = Image.open('QH2.jpg').convert('L')
l = np.asarray(im)

distx = 300 - 278
disty = 177 - 114
plt.plot(range(278, 278+distx), [175 for _ in range(distx)], color='k')
plt.plot([300 for _ in range(disty)], range(114, 114+disty), color='k')

print 'distx', distx, 'disty', disty
print 'r:', float(distx)/disty

plt.imshow(l)
plt.show()

