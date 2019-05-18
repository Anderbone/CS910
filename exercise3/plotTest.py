import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 0.7, 0.01)
y1 = 4.57 * x - 1.04
y2 = -3.3556*x + 10.4968*x**2 + 0.3477
y3 = 10.3377 * x**3
y4 = 10**(2.7183*x - 1.2596)

plt.plot(x,y1,color="red",linewidth=2,label='fangcheng')#方程
plt.plot(x,y2,color="blue",label='qiexian')
plt.plot(x,y3,color="green",label='qiexian')
plt.plot(x,y4,color="black",label='qiexian')
plt.show()