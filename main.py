import random
import math
import numpy as np
import matplotlib.pyplot as plt

n = 14
omega = 1800
N = 256
w = 0
i = 0
t = 0
x = np.zeros(N)
y = np.zeros(N)


def fun1(i, n, w, omega, arr):
    while i < n:
        w += omega / n
        i += 1
        t = 0
        A = random.random()
        fi = np.random.uniform(-np.pi / 2, np.pi / 2)
        while t < N:
            arr[t] += A * math.sin(w * t + fi)
            t += 1


fun1(0, n, w, omega, x)
fun1(0, n + 20, w, omega, y)

plt.plot(x)
plt.ylabel('x(t)')
plt.show()

plt.plot(y)
plt.ylabel('y(t)')
plt.show()

print("Мат Ожидание")
print(np.average(x))  # Мат Ожидание
print("Дисперсия")
print(np.std(x) ** 2)  # Дисперсия

# My method
XXcor1 = np.zeros(N)
XYcor1 = np.zeros(N)


def corr(arr1, arr2, result_arr):
    newarr2 = np.zeros(N * 2)
    i = 0
    while i < N:
        newarr2[i] = arr2[i]
        i += 1
    tau = 0
    while tau < N:
        iterator = 0
        while iterator < N:
            result_arr[tau] += (arr1[iterator] - np.average(arr1)) * (newarr2[iterator + tau] - np.average(arr2))
            iterator += 1
        result_arr[tau] = (result_arr[tau] / (N-1))
        tau += 1
    print(result_arr)


corr(x, x, XXcor1)
corr(x, y, XYcor1)

XXcor1 = XXcor1 / (np.std(x) ** 2)
XYcor1 = XYcor1 / (np.std(x) * np.std(y))

#  Numpy method
a = (x - np.mean(x)) / (np.std(x) * len(x))
a2 = (x - np.mean(x)) / (np.std(x))
b = (y - np.mean(y)) / (np.std(y))

XXcor2 = np.correlate(a, a2, 'full')
XYcor2 = np.correlate(a, b, 'full')

# Drawing
plt.plot(XXcor1)
plt.ylabel('Автокореляция1')
plt.show()

plt.plot(XYcor1)
plt.ylabel('Взаимная кореляция1')
plt.show()

plt.plot(XXcor2)
plt.ylabel('Автокореляция2')
plt.show()

plt.plot(XYcor2)
plt.ylabel('Взаимная кореляция2')
plt.show()
