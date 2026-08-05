import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0, 2*np.pi)
sin = np.sin(x)
cos = np.cos(x)


plt.plot(x, sin)
plt.plot(x, cos)

plt.savefig("sincos.png")

plt.close()