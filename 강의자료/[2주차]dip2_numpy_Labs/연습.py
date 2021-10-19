import numpy as np

a = np.arange(5).reshape(5)             # 1차원 1x4 행렬 하나
print(a)
print('==============================')

b = np.arange(5).reshape(1, 5)          # 1차원 1x4 행렬 하나 2차원.
print(b)
print('==============================')

c = np.arange(24).reshape(2, 4, 3)      # 2차원 2개에 4x3 행렬 하나씩 3차원
print(c)
print('==============================')

a = np.arange(5)
b = np.arange(5, 10)
print(a*b)      # 원소끼리의 곱
print('==============================')
print(5*a)
print(b)
print('==============================')
b = a.copy()
print(b)
print('==============================')
print(a)
a = np.clip(a, 2, 3)
print(a)
print('==============================')
