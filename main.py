import random
import math
import time

import numpy as np
import matplotlib.pyplot as plt

n = 14
omega = 1800
N = 256  # 256
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


def fft(x):
    V = len(x)
    if V <= 1:
        return x
    even = fft(x[0::2])
    odd = fft(x[1::2])
    T = [np.exp(-2j * np.pi * m / V) * odd[m] for m in range(V // 2)]
    return [even[m] + T[m] for m in range(V // 2)] + [even[m] - T[m] for m in range(V // 2)]


A = np.zeros(N)
B = np.zeros(N)
y = fft(x)
A = np.abs(f)
B = np.abs(y)
A = 2 * A / N
B = 2 * B / N

plt.plot(A)
plt.ylabel("Фурье")
plt.xlim(0, 128)  # 128
plt.show()

plt.plot(B)
plt.ylabel("fft")
plt.xlim(0, 128)  # 128
plt.show()