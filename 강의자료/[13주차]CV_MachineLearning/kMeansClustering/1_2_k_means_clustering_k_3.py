"""
K-means clustering algorithm applied to three different 'clusters' of points (k=3)

N=1000개의 데이터에 대해 attempts=1일 경우
<class 'numpy.ndarray'> 3000 (3000, 2)
center: [[ 24.245     24.559002]
 [ 74.34901   74.451004]
 [124.269005 124.70801 ]]

N=1000개의 데이터에 대해 attempts=10일 경우
<class 'numpy.ndarray'> 3000 (3000, 2)
center: [[ 24.618002  24.598001]
 [ 74.803     73.806   ]
 [124.644005 124.993004]]

N=1000개의 데이터에 대해 attempts=20일 경우
<class 'numpy.ndarray'> 3000 (3000, 2)
center: [[ 25.035002  25.129002]
 [ 74.253006  74.604004]
 [124.828    125.383   ]]

"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Create data (three different 'clusters' of points (it should be of np.float32 data type):

N = 50     # 원본의 경우
data = np.float32(np.vstack(
    (np.random.randint(0, 40, (N, 2)), np.random.randint(30, 70, (N, 2)), np.random.randint(60, 100, (N, 2)))))
    #(np.random.randint(0, 50, (N, 2)), np.random.randint(30, 80, (N, 2)), np.random.randint(60, 100, (N, 2)))))

#N = 150    # center가 어느 곳일지 짐작이 되는 사례임. center=[ [25, 25] [75 75] [125] [125] ] N이 클 수록 더 근접하게 될 것이다.
#data = np.float32(np.vstack(
#    (np.random.randint(0, 40, (N, 2)), np.random.randint(50, 80, (N, 2)), np.random.randint(100, 150, (N, 2)))))

print(f'data: {type(data)}, shape={data.shape}, len={len(data)}')

# Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
# In this case the maximum number of iterations is set to 20 and epsilon = 1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

# Apply kmeans algorithm
# K = 3, which is the number of clusters to split the set by
# attempts = 10, which specifies the number of times the algorithm is executed using different initial labellings
# (the algorithm returns the labels that yield the best compactness)
# Flag cv2.KMEANS_RANDOM_CENTERS selects random initial centers in each attempt. You can also use cv2.KMEANS_PP_CENTERS
ret, label, center = cv2.kmeans(data, 3, None, criteria,
                                10, cv2.KMEANS_RANDOM_CENTERS)

print(f'ret: {ret:#.2f}')
#num_data = len(data); k=3; print(f'ret: {ret:#.2f}, normalized compactness={ret/(num_data*k):#.2f}')
print(f'label: type={type(label)}, shape={label.shape}')            # label: type=<class 'numpy.ndarray'>, shape=(150, 1)
print(f'center: type={type(center)}, shape={center.shape}')         # center: type=<class 'numpy.ndarray'>, shape=(2, 2)
print('center:\n', center)


# Now separate the data using label output (stores the cluster indices for every sample)
# Therefore, we split the data to different clusters depending on their labels:
A = data[label.ravel() == 0]
B = data[label.ravel() == 1]
C = data[label.ravel() == 2]

print(f'A: shape={A.shape}')
print(f'B: shape={B.shape}')
print(f'C: shape={C.shape}')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means clustering algorithm", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the 'original' data:
ax = plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data")

# Plot the 'clustered' data and the centroids
ax = plt.subplot(1, 2, 2)
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')
plt.scatter(C[:, 0], C[:, 1], c='r')
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 3)")

# Show the Figure:
plt.show()
