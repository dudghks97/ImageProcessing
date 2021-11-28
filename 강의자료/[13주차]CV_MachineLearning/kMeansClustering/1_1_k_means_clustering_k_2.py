"""
K-means clustering algorithm applied to three different 'clusters' of points (k=2)
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create data (three different 'clusters' of points (it should be of np.float32 data type):
data = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))))
print(f'1. data: {type(data)}, shape={data.shape}, len={len(data)}')   # data: <class 'numpy.ndarray'>, shape=(150, 2), len=150

# Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
# In this case the maximum number of iterations is set to 20 and epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 20, 1.0)

# Apply k-means algorithm
# cv.kmeans() 함수
#   https://docs.opencv.org/4.5.4/d5/d38/group__core__cluster.html#ga9a34dc06c6ec9460e90860f15bcd2f88
#  retval, bestLabels, centers = cv.kmeans(data, K, bestLabels, criteria, attempts, flags[, centers])
#
#   K, which is the number of clusters to split the set by
#   attempts = 10, which specifies the number of times the algorithm is executed
#       using different initial labellings
#       (the algorithm returns the labels that yield the best compactness):
#       이 만큼의 시도를 해본 후 가장 적은 compactness를 내는 클러스터링 결과를 반환한다.
#   flags: 클러스터의 초기 선정을 어떻게 할 것인가를 결정
#       cv2.KMEANS_RANDOM_CENTERS selects random initial centers in each attempt.
#       cv2.KMEANS_PP_CENTERS: kmeans++ center initialization by Arthu 알고리즘 사용.
#       cv2.KMEANS_USE_INITIAL_LABELS: 프로그래머가 초기 위치를 지정. 이후 랜덤.
#       cv2.KMEANS_*_CENTERS: 선정 알고리즘을 프로그래머가 지정
# ret: compactness=sum of all (sample - center)**2
# bestLabels: 해당 샘플이 어느 군집에 속한다고 판단하는 것이 좋은지에 대한 라벨링.
#   해당 샘플과 똑 같은 길이의 라벨 자료는 (샘플의 수 x 1)의 차원을 갖는다.
#   라벨 자료는 샘플과 같은 순서로 배열되어 있다.
#   라벨은 K=2이면 라벨은 0, 1의 값을 갖는다.
#   라벨은 K=3이면 라벨은 0, 1, 2의 값을 갖는다.
#   K가 커진다고 해서 라벨의 차원이 달라질 것은 없다. 라벨데이터는 int32형으로 관리된다.
# center: K개의 클러스터의 중심점. K개 만큼 필요한데, 각 중심점이 (x, y) 좌표를 가지므로 (k, 2)의 차원을 갖는다.
#
ret, label, center = cv2.kmeans(data, 2, None, criteria,
                                10, cv2.KMEANS_RANDOM_CENTERS)
print(f'2. ret, compactness={ret:#.2f}')                                           # ret: 65391.29
#print(f'ret: {ret:#.2f}, normalized compactness={ret/(50*3 * 2):#.2f}')
# 2. ret, compactness=68776.15

print(f'3. label: type={type(label)}, shape={label.shape}, dtype={label.dtype}')
# 3. label: type=<class 'numpy.ndarray'>, shape=(150, 1), dtype=int32

print(f'4. center: type={type(center)}, shape={center.shape}')
# 4. center: type=<class 'numpy.ndarray'>, shape=(2, 2)

print('5. center:\n', center)
# 5. center:
#  [[65.639534 71.523254]
#  [21.390625 25.171875]]

# Now separate the data using label output (stores the cluster indices for every sample)
# Therefore, we split the data to different clusters depending on their labels:
# ravel() 함수를 이용해서 데이터 샘플을 1차원 한줄로 나열한다. 총 150개이므로 shape=(150,)
print(f'6. label.ravel(): shape={label.ravel().shape}')
# 6. label.ravel(): shape=(150,)

# K=2이므로 2개의 그룹 A, B로 라벨링을 한다. 0 혹은 1로 lavel이 부여된다.
A = data[label.ravel() == 0]    # data[150,2] 중에서 첫번째 원소에 대해 label 값이 0인 원소만 A에 배정
B = data[label.ravel() == 1]

# 각 그룹의 shape를 살펴본다.
# 그중, row는 합해서 샘플의 총 갯수가 되어야 한다.
# 랜덤하게 샘플데이터가 자리잡기 때문에 실험을 할 때마다 그룹의 길이가 달라질 것이다.
print(f'7. A: shape={A.shape}')    # A: shape=(78, 2)
print(f'8. B: shape={B.shape}')    # B: shape=(72, 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the 'original' data:
ax = plt.subplot(1, 2, 1)

# scatter() 함수는 (x, y) 좌표를 사용한다.
# x 좌표=data[:, 0], y좌표=data[:, 1]
plt.scatter(data[:, 0], data[:, 1], c='c')

plt.title("data")

# Plot the 'clustered' data and the centroids
ax = plt.subplot(1, 2, 2)

# 그룹 A, B의 해당 좌표에 blue, green으로 점을 찍는다.
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')

# 각 그룹의 중심점에 magenta 색상으로 squre를 크기 100으로 마킹한다.
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 2)")

# Show the Figure:
plt.show()
