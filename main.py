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
f = np.complex64(np.zeros(N))


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

plt.plot(x)
plt.ylabel('x(t)')
plt.show()

p = 0
k = 0

while p < N:
    k = 0
    while k < N:
        f[p] += x[k] * (np.cos(2 * np.pi * p * k / N) - np.sin(2 * np.pi * p * k / N) * 1j)
        k += 1
    p += 1

print(f)
A = np.zeros(N)

A = np.abs(f)
A = 2 * A / N
print(A)

H = list(range(1, 128))

plt.plot(A)
plt.ylabel("Фурье")
plt.xlim(0, 128)
plt.show()