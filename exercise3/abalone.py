import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy
import math

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

def f_2(x, A, B, C):
    return A*x + B*x*x + C

def f_3(x, A):
    return A*x*x*x

def f_4(x, A, B):
    # return pow(10, A*x + B)
    # return np.power(math.e, A*x+B)
    return np.exp(A*x+B)

def f_5(x, A, B, C ,D):
    return A*x*x*x + B*x*x + C*x +D

def f_6(x,A,B):
    return A*x*x*x+B

def f_7(x,A,B):
    return A*np.exp(x)+B


def conf(ylist, flist):
    residuals = list(map(lambda x: (x[0] - x[1]) ** 2, zip(ylist, flist)))
    ss_res = numpy.sum(residuals)
    # print(ss_res)
    tot = list((x - numpy.mean(ylist)) ** 2 for x in ylist)
    ss_tot = numpy.sum(tot)
    r_squared = (1 - (ss_res / ss_tot)) ** 0.5
    print('linear  def cof')
    print(r_squared)
    return r_squared #actually it's r. not r squared



# Polynomial Regression
def polyfit(x, y, degree):
    results = {}

    coeffs = numpy.polyfit(x, y, degree)

     # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()

    # r-squared
    p = numpy.poly1d(coeffs)
    # fit values, and mean
    yhat = p(x)                         # or [p(z) for z in x]
    ybar = numpy.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = numpy.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = numpy.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results['cof'] = (ssreg / sstot)**0.5

    return results
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

# # exercise2
# plt.scatter(height_list, length_list)
#
# A1, B1 = optimize.curve_fit(f_1, height_list, length_list)[0]
# x1 = np.arange(0, 0.4, 0.01)
# y1 = A1 * x1 + B1
# plt.plot(x1, y1, "red")
#
# plt.show()
# co = np.corrcoef([length_list, diameter_list, height_list, whole_list, shucked_list, viscera_list, shell_list])
# print(co)

# xm = np.sort(man_ring)
# ym = np.arange(len(xm))/float(len(xm))
# plt.plot(xm, ym)
# plt.subplot(3,1,1)
# plt.title('M_ring CDF')
# plt.xlabel('ring')
# plt.plot(np.sort(man_ring), np.linspace(0, 1, len(man_ring), endpoint=False))
#
# plt.subplot(3,1,2)
# plt.title('F')
# plt.plot(np.sort(fem_ring), np.linspace(0, 1, len(fem_ring), endpoint=False))
#
# plt.subplot(3,1,3)
# plt.title('I')
# plt.plot(np.sort(inf_ring), np.linspace(0, 1, len(inf_ring), endpoint=False))

# plt.show()
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
# # exercise3.1
# plt.scatter(length_list, diameter_list)
#
# A1, B1 = optimize.curve_fit(f_1, length_list, diameter_list)[0]
# x1 = np.arange(0, 0.8, 0.01)
# y1 = A1 * x1 + B1
# plt.xlabel('Length')
# plt.ylabel('Diameter')
# my_x_ticks = np.arange(0, 1, 0.1)
# my_y_ticks = np.arange(0, 1, 0.1)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# print(A1, B1)
# plt.plot(x1, y1, "red")
# plt.show()
# 3.3
plt.subplot(2,2,1)
plt.scatter(diameter_list, whole_list)

A1, B1 = curve_fit(f_1, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * x1 + B1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1, B1)
plt.plot(x1, y1, "red")

flist = []
for i in diameter_list:
    flist.append(f_1(i, A1, B1))
conf(whole_list, flist)
# # popt, pcov = curve_fit(f_1, diameter_list, whole_list)
# # for whole, y in whole_list,ylist:
# #     residuals = whole - y
# #     ss_res = numpy.sum(residuals**2)
# #     ss_tot = numpy.sum((whole-numpy.mean(whole))**2)
# # residuals = whole_list - ylist
# residuals = list(map(lambda x: (x[0] - x[1])**2, zip(whole_list, ylist)))
# ss_res = numpy.sum(residuals)
# print(ss_res)
# tot = list((x - numpy.mean(whole_list))**2 for x in whole_list)
# ss_tot = numpy.sum(tot)
# r_squared = (1 - (ss_res / ss_tot)) ** 0.5
# print('linear cof')
# print(r_squared)


plt.subplot(2,2,2)
plt.scatter(diameter_list, whole_list)

A1, B1 ,C1= curve_fit(f_2, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * x1 + B1*x1*x1+C1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1, B1, C1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_2(i, A1, B1,C1))
conf(whole_list, flist)

plt.subplot(2,2,3)
plt.scatter(diameter_list, whole_list)
A1= curve_fit(f_3, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * x1*x1*x1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_3(i, A1))
print('A1*x1*x1*x1')
conf(whole_list, flist)
print('-------------------------------------log start')
plt.subplot(2,2,4)
plt.scatter(diameter_list, whole_list)
A1, B1 = curve_fit(f_1, diameter_list, np.log(whole_list))[0]
print(A1,B1)
x1 = np.arange(0, 0.7, 0.01)
y1 = np.exp(A1*x1+B1)
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print('log para')
print(A1, B1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_1(i, A1, B1))
print('most important , fourth one cof')
conf(np.log(whole_list), flist)
print('---------------------------------log over')

plt.show()
# coeffs = np.polyfit(diameter_list, whole_list, 2)
# print(coeffs)
a = polyfit(diameter_list, whole_list, 1)
print(a)
a = polyfit(diameter_list, whole_list, 2)
print(a)
a = polyfit(diameter_list, whole_list, 3)
print(a)


A1, B1 ,C1, D1= curve_fit(f_5, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * x1 *x1*x1 + B1*x1*x1+C1*x1+D1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1, B1, C1,D1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_5(i, A1, B1,C1,D1))
print('A1 * x1 *x1*x1 + B1*x1*x1+C1*x1+D1')
conf(whole_list, flist)

A1, B1= curve_fit(f_6, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * x1 *x1*x1 + B1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1, B1, C1,D1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_6(i, A1, B1))
print('A1 * x1 *x1*x1 + B1')
conf(whole_list, flist)

A1, B1= curve_fit(f_7, diameter_list, whole_list)[0]
x1 = np.arange(0, 0.7, 0.01)
y1 = A1 * np.exp(x1)+B1
plt.xlabel('Diameter')
plt.ylabel('Whole weight')
my_x_ticks = np.arange(0, 3, 0.1)
my_y_ticks = np.arange(0, 3.5, 0.5)
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
ax = plt.gca()
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
print(A1, B1, C1,D1)
plt.plot(x1, y1, "red")
flist = []
for i in diameter_list:
    flist.append(f_7(i, A1, B1))
print(A1,B1)
print('expppp')
conf(whole_list, flist)