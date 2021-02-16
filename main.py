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

while i < n:
    w += omega / n
    i += 1
    t = 0
    A = random.random()
    fi = np.random.uniform(-np.pi / 2, np.pi / 2)
    while t < N:
        # x.append(A * math.sin(w * t + fi))
        x[t] += A * math.sin(w * t + fi)
        t += 1

plt.plot(x)
plt.ylabel('x(t)')
plt.show()
print(np.average(x))
print(np.std(x) ** 2)
