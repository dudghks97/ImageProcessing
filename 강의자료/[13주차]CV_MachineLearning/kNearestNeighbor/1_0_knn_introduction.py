"""
Simple introduction to k-Nearest Neighbour (k-NN) algorithm with OpenCV

개요:
    본 프로그램은 k-Nearest Neighbour의 활용 기법을 연마하기기 위한 것이다.
    임의의 테스트 데이터 1개가 어느 그룹에 속하는지를 k-nn 알고리즘(k=3)을 이용해 판단한 결과를 보인다.
절차:
    1) 임의로 생성한 16개의 (x,y) 좌표 데이터를 임의로 2개의 그룹으로 나누어 학습 데이터를 생성한다.
       랜덤하게 배정된 labels를 이용하여 학습 데이터를 생성하였다.
    2) test data를 1개 랜덤하게 생성한다. 이 데이터가 어느 그룹에 속하는지를 k-nn 방법으로 분류할 예정이다.
    3) knn 학습 모델을 구축한다.
    4) test 데이터가 어느 그룹에 속하는지 k-nn으로 분류한다.
    5) 위 결과를 그림으로 표현한다.

"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# The data is composed of L points:
L = 16  # 학습 데이터의 수
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)

# We create the labels (0: red, 1: blue) for each of the 16 points:
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)

# Create the sample point to be classified:
N = 1   # 테스트 데이터의 수
sample = np.random.randint(0, 100, (N, 2)).astype(np.float32)

# k-NN creation:
knn = cv2.ml.KNearest_create()
# k-NN training:
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
# k-NN find nearest:
k = 3
ret, results, neighbours, dist = knn.findNearest(sample, k)

# Plot all the data and print the results:
# Create the dimensions of the figure:
fig = plt.figure(figsize=(8, 6))
#fig.patch.set_facecolor('silver')

# Take points with label 0 (will be the red triangles) and plot them:
red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')

# Take points with label 1 (will be the blue squares) and plot them:
blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')

# Plot the sample point:
plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')

# Print results:
print(f'\nnum of train data={L}, num of test data={N}, k={k}')
print(f"result: {results}, shape={results.shape}")
print(f"neighbours: {neighbours}, shape={neighbours.shape}")
print(f"distance: {dist}, shape={dist.shape}")

# Set the title:
if results[0][0] > 0:
    plt.suptitle("k-NN algorithm: sample green point is classified as blue (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')
else:
    plt.suptitle("k-NN algorithm: sample green point is classified as red (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')

# Show the results:
plt.show()
