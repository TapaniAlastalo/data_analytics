import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(-2,10,1000)
y1 =  2*x +1
y2 = 0.2*x**2-2
y3 = (x)**0.5
fig, ax = plt.subplots()

plt.plot(x, y1, 'r-', label='2x+1')
plt.plot(x, y2, 'b-.', label='0,2x^2-3', linewidth=3.0)
plt.plot(x, y3, 'k-', label='neliöjuuri')
plt.legend(loc='lower right')

ax.set_facecolor((0.0, 1.0, 0.3))

ax.axhline(y=0, color='k', linewidth=4.0)
ax.axvline(x=0, color='k', linewidth=4.0)


plt.xlim(-2, 10)
plt.xticks([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.ylim(-4, 12)
plt.yticks([-4, -2, 0, 2, 4, 6, 8, 10])
plt.grid(True, which='both', c='black', linestyle='--')

plt.show()