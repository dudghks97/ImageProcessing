"""
K-means clustering data visualization (introduction to k-means clustering algorithm)
개요
    30개의 랜덤 좌표 x 3세트,총 150개의 랜덤 좌표에 magenta 색상으로 표시한다.
    각 세트는 지정된 범위의 랜덤 좌표(x, y)로 구성되어 있다.
    랜덤 좌표 정보는 randint() 함수로, 점을 찍는 함수는 scatter()로 구현하였다.


"""


# Import required packages:
import numpy as np
from matplotlib import pyplot as plt

"""
# 시작하기에 앞서... scatter 함수 연습..
# scatter(x, y, c) 함수에서 (x, y) 좌표 연습용. scatter() 함수에서 data2를 출력해 보시오.
x = np.array([[i] for i in range(30)])      # x축 좌표: (30x1). 0~29
print('type(x)=', type(x), x.shape)
y = 5 * np.ones((30, 1))                    # y축 좌표: (30x1). 모두 5
print('type(y)=', type(y), y.shape)
data2 = np.float32(np.hstack((x, y)))       # (x, y) 좌표
print('type(data2)=', type(data2), data2.shape)

# (x,y) 순으로 지정한다. m=magenta, c=cyan
# s: marker size in points**2
# maker:  default 'o', circle. 's'=square, '+'=plus
plt.subplot(121)
plt.scatter(data2[:, 0], data2[:, 1], s=20, marker='s', c='y')
plt.title("data to be clustered")

plt.subplot(122)
#plt.scatter(data2[:, 0], data2[:, 1], s=20, marker='o', c='b')
clr_array1 = np.array([range(30)])
plt.scatter(data2[:, 0], data2[:, 1], s=20, marker='o', c=clr_array1)

plt.scatter(data2[:, 0], data2[:, 1]*2, s=20, marker='+', c='m')

# Show the Figure:
plt.show()

exit()
"""





# numpy.random.randint() - 특정 범위의 정수로 이루어진  특정 차원의 어레이 생성하기
# numpy.random.randint(low, high=None, size=None, dtype='l')
#   Return random integers from low (inclusive) to high (exclusive).
#   Return random integers from the “discrete uniform” distribution of the specified dtype in the “half-open” interval [low, high).
#   If high is None (the default), then results are from [0, low).

# Plot the 'original' data:
# scatter(x, y, s=None, c=None,..)
# x, y : array_like, shape (n, )




# Create data (three different 'clusters' of points (it should be of np.float32 data type):
# (x, y) 좌표를 의미하는 2D 데이터 포인트를 50개씩 선언한 것을 column 방향으로 묶는다.
# 첫 번째 스택 분석 사례: np.random.randint(0, 40, (50, 2))
#   위치값이 [0, 40)까지의 좌표 값이 존재하는 50개의 2차원 좌표 어레이를 생성한다.
#data = np.float32(np.vstack(
#    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))))

a = np.random.randint(0, 40, (50, 2))       # a의 값의 범위는 0~39. 이런 값들도 이루어진 좌표 데이터 50개.
#print('type(a)=', type(a), a.shape)        # type(a)= <class 'numpy.ndarray'> (50, 2)
b = np.random.randint(30, 70, (50, 2))      # b의 값의 범위는 30~69. 이런 값들도 이루어진 좌표 데이터 50개
c = np.random.randint(60, 100, (50, 2))     # c의 값의 범위는 60~100. 이런 값들도 이루어진 좌표 데이터 50개

data = np.float32(np.vstack((a, b, c)))     # a. b, c의 좌표로 150개의 좌표 데이터
#print('type(data)=', type(data), data.shape) # type(data)= <class 'numpy.ndarray'> (150, 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(6, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the 'original' data:
# scatter(x, y, s=None, c=None,..)
# x, y : array_like, shape (n, )
#
ax = plt.subplot(1, 1, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
#plt.scatter(data2[:, 0], data2[:, 1], c='y')      # (x,y) 순으로 지정한다. m=magenta, c=cyan
plt.title("data to be clustered")

# Show the Figure:
plt.show()
