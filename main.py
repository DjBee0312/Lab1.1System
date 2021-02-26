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

print(np.average(x))  # Мат Ожидание
print(np.std(x) ** 2)  # Дисперсия

a = (x - np.mean(x)) / (np.std(x) * len(x))
a2 = (x - np.mean(x)) / (np.std(x))
b = (y - np.mean(y)) / (np.std(y))
XXcor = np.correlate(a, a2, 'full')
XYcor = np.correlate(a, b, 'full')

plt.plot(XXcor)
plt.ylabel('Автокореляция')
plt.show()

plt.plot(XYcor)
plt.ylabel('Взаимная кореляция')
plt.show()
