import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

length_list = []
diameter_list = []
height_list = []
whole_list = []
shucked_list = []
viscera_list = []
shell_list = []
man_ring = []
fem_ring = []
inf_ring = []

def f_1(x,A,B):
    return A*x + B


with open('abalone.data', 'r') as f:
    for line in f:
        line = line.split(',')
        length_list.append(float(line[1]))
        diameter_list.append(float(line[2]))
        height_list.append(float(line[3]))
        whole_list.append(float(line[4]))
        shucked_list.append(float(line[5]))
        viscera_list.append(float(line[6]))
        shell_list.append(float(line[7]))
        if line[0] == 'M':
            man_ring.append(float(line[8]))
        elif line[0] == 'F':
            fem_ring.append(float(line[8]))
        elif line[0] == "I":
            inf_ring.append(float(line[8]))


plt.scatter(height_list, length_list)

A1, B1 = optimize.curve_fit(f_1, height_list, length_list)[0]
x1 = np.arange(0, 0.4, 0.01)
y1 = A1 * x1 + B1
plt.plot(x1, y1, "red")

plt.show()

co = np.corrcoef([length_list, diameter_list, height_list, whole_list, shucked_list, viscera_list, shell_list])
print(co)

# xm = np.sort(man_ring)
# ym = np.arange(len(xm))/float(len(xm))
# plt.plot(xm, ym)
plt.subplot(3,1,1)
plt.title('M_ring CDF')
plt.xlabel('ring')
plt.plot(np.sort(man_ring), np.linspace(0, 1, len(man_ring), endpoint=False))

plt.subplot(3,1,2)
plt.title('F')
plt.plot(np.sort(fem_ring), np.linspace(0, 1, len(fem_ring), endpoint=False))

plt.subplot(3,1,3)
plt.title('I')
plt.plot(np.sort(inf_ring), np.linspace(0, 1, len(inf_ring), endpoint=False))

plt.show()
# plt.xlim(xmax=0.25, xmin=0)
# plt.ylim(ymax=0.880, ymin=0.100,)
# plt.xticks([x*3 for x in range(10)])
# plt.yticks([y*5 for y in range(30)])

#
# ans = list(zip(height_list,length_list))
# ans.sort()
#
# for point in ans:
#     plt.scatter(point[0], point[1])
#
# plt.xticks([x*3 for x in range(10)])
# plt.yticks([y*5 for y in range(30)])