"""
Simple introduction to k-Nearest Neighbour (k-NN) algorithm with OpenCV

제목: 랜덤 데이터와 레이블에 기반한 knn모델의 분류 시연 프로그램

개요:
    본 프로그램은 k-Nearest Neighbour의 활용 기법을 연마하기기 위해 학습 모델이 있다고 가정하고,
    k-nn 알고리즘이 k에 따라 어떻게 다른 결과를 나타낼 수 있는지 보이고자 한다.
    본 예제의 상황은 데이터의 레이블이 랜덤으로 설정되었기 때문에 근접한 데이터라고 해도 사실 전혀 상관이 없다.
    이 때문에 k를 크게 하면 할 수록 받아들이기 어려운 결과를 낼 수 밖에 없다.

절차:
    단계 1: 학습 데이터를 만든다. 랜덤한 값(x, y)으로 데이터를 L개 생성한다.
	    이들을 2개의 그룹으로 랜덤하게 나눈다.
    단계 2: knn 학습 모델을 구축한다.
    단계 3: test data를 N개 생성한다.
    단계 4: test 데이터가 어느 그룹에 속하는지 K를 바꾸어 가며 k-nn으로 분류한다.
	    이때 학습 데이터와 테스트 데이터 및 분류 결과를 화면에 한 화면에 보인다.
	    이후 k를 바꾸어 가면 같이 실행한 결과를 보인다.

미션:
    k=1에 대해 test 데이터를 중심으로 가장 가까운 학습 데이터를 만나는 원을 그려보자.
    k=2에 대해 test 데이터를 중심으로 가장 가까운 학습 데이터를 만나는 2개의 원을 해당 데이터의 색상으로 그려보자.

질문:
    k=1일 때 모든 test 데이터의 dist는 0일까?


retval, results, neighborResponses, dist = cv.ml_KNearest.findNearest( samples, k[, results[, neighborResponses[, dist]]] )
    samples: Input samples stored by rows. It is a single-precision floating-point matrix of <number_of_samples> * k size.
    k: Number of used nearest neighbors. Should be greater than 1.
    results: Vector with results of prediction (regression or classification) for each input sample.
            It is a single-precision floating-point vector with <number_of_samples> elements.
    neighborResponses: Optional output values for corresponding neighbors.
            It is a single- precision floating-point matrix of <number_of_samples> * k size.
    dist: Optional output distances from the input vectors to the corresponding neighbors.
        It is a single-precision floating-point matrix of <number_of_samples> * k size.


"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# 단계 1: 학습 데이터를 만든다. 랜덤한 값(x, y)으로 데이터를 L개 생성한다.
# 이들을 2개의 그룹으로 랜덤하게 나눈다.
# --------------------------------------------------------------------------------------------
# The data is composed of L points: [0, 100)의 범위를 가진 L개의 점(2차원: x, y) 데이터 정의
# train data로 쓰인다.
L = 16
print(f'\nStep 1: L={L}, Making random training data & labels..')
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
print('data for training:', type(data), data.shape)

# We create the labels (0: red, 1: blue) for each of the L points:
# 각 점에 대해 0 혹은 1의 레이블을 할당. train을 위한 label로 쓰인다.
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)
print('Labels for training:', type(labels), labels.shape)


# --------------------------------------------------------------------------------------------
# 단계 2: knn 학습 모델을 구축한다.
# --------------------------------------------------------------------------------------------
# k-NN creation:
print('\nStep 2: Creating & training a knn model..')
knn = cv2.ml.KNearest_create()
# k-NN training: 학습용 데이터, data와 이를 지도학습 시킬 수 있는 label이 필요하다.
knn.train(data, cv2.ml.ROW_SAMPLE, labels)


# --------------------------------------------------------------------------------------------
# 단계 3: test data를 N개 생성한다.
# Create the sample point to be classified:
# --------------------------------------------------------------------------------------------
N = 8       # 샘플 포인트의 개수
print(f'\nStep 4: N={N}, Making random test data ..')
sample = np.random.randint(0, 100, (N, 2)).astype(np.float32)
print('sample:', type(sample), sample.shape)


# --------------------------------------------------------------------------------------------
# 단계 4: test 데이터가 어느 그룹에 속하는지 K를 바꾸어 가며 k-nn으로 분류한다.
# 이때 학습 데이터와 테스트 데이터 및 분류 결과를 화면에 한 화면에 보인다.
# 이후 k를 바꾸어 가면 같이 실행한 결과를 보인다.
# --------------------------------------------------------------------------------------------

#from collections import Counter

def print_ret_values(ret_val, rslt_v='no', nfg_v='no', dst_v='no'):
    """findNearest() 함수가 반환하는 반환 값을 이해하기 위해 반환값을 출력하는 함수"""
    retval, results, neighbours, dist = ret_val
    #print('retval:', type(retval), retval)   # 모드 k 에 대해 0이 나옴. 삭제..
    print('results:', type(results), results.shape)
    if rslt_v != 'no': print(results)
    print('neighbours:', type(neighbours), neighbours.shape)
    if nfg_v != 'no': print(neighbours)
    print('dist:', type(dist), dist.shape)
    if dst_v != 'no': print(dist)

print(f'\nStep 5: L={L}, N={N}, Testing the test data  ..')

# 실험에 적용할 K 값의 리스트
K = [1, 3, 5]
K = [1]
K = [1, 3]

for k in K:
    print(f'\ntest=sample: L={L}, N={N}, k={k}-------')
    ret_val = knn.findNearest(sample, k)
    #print_ret_values(ret_val, rslt_v='yes', nfg_v='yes', dst_v='yes')
    #print_ret_values(ret_val, 'yes', 'yes', 'yes')     # 타입, shape와 값 모두 출력한다.
    print_ret_values(ret_val)   # 값은 출력하지 않는다. 3개의 반환 값에 대한 타입, shape만 출력한다.
    ret, results, neighbours, dist = ret_val

    # 그룹핑한 결과를 그림으로 표현한다. ------

    # Plot all the data and print the results:
    # Create the dimensions of the figure:
    fig = plt.figure(figsize=(8, 6))
    #fig.patch.set_facecolor('silver')
    # Take points with label 0 (will be the red triangles) and plot them:
    red_triangles = data[labels.ravel() == 0]
    plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')    # 적색(r), 삼각형(^), 크기=200

    # Take points with label 1 (will be the blue squares) and plot them:
    blue_squares = data[labels.ravel() == 1]
    plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')

    # Plot the sample point:
    plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')


    for i in range(N):
        if results[i][0] > 0:   # 판정이 1(blue 사각형)이라면 blue 색상으로 테스트 데이터를 적어 넣는다.
            #print(f"sample {i} at location[{sample[i]}] = blue")
            plt.text(sample[i, 0], sample[i, 1], str(i), color='b', fontsize=30)    # 해당 번호의 샘플은 blue(category 0)에 속함.
        else:       # 판정이 0(red 삼각형)이라면 red 색상으로 테스트 데이터를 적어 넣는다.
            #print(f"sample {i} at location[{sample[i]}] = red")
            plt.text(sample[i, 0], sample[i, 1], str(i), color='r', fontsize=30)    # 해당 번호의 샘플은 red(category 1)에 속함.

    plt.title(f'k={k}: label 0=red triangle, 1=blue rectangle')

plt.show()
