"""
Handwritten digits recognition using KNN and raw pixels as features
Understanding k-Nearest Neighbour in OpenCV document
    https://docs.opencv.org/3.4/d5/d26/tutorial_py_knn_understanding.html

개요
    '../data/digits.png' 파일에서 5000개의 손글씨(20x20) 숫자 0~9를 읽어들여
    이것의 반은 KNN 학습에 사용하고, 나머지는 분류 정확성을 판단하는 데 사용한다.

주요 사항: KNN 학습모델을 만들고, 분류작업을 하기 위해서는 5단계부터 사용되는 다음 3가지의 함수 활용이 필요하다.
        knn = cv2.ml.KNearest_create()
        knn.train(학습데이터, cv2.ml.ROW_SAMPLE, 학습_데이터_라벨)
        knn.findNearest(테스트_데이터, k)

처리 절차
    1. 영상 파일을 읽어들이고 라벨링을 행한다.
        digits.png에는 20x20으로 이루어진 손글씨 폰트가 100x50(가로x세로)개 나열되어 있다. 레이블(0~9)은 프로그램으로 자체 생성한다.
        영상파일을 가로로 자른 후 다시 세로로 잘라 (5000, 20, 20)의 어레이 변수에 저장한다.
        digits.png에는 20x20으로 이루어진 손글씨 폰트가 100x50(가로x세로)개 나열되어 있다. 레이블(0~9)은 프로그램으로 자체 생성한다.
        영상파일을 가로로 자른 후 다시 세로로 잘라 (5000, 20, 20)의 어레이 변수에 저장한다.

    2~4. 학습에 적합한 데이터로 정렬한다.
        2. 학습을 위해 영상 데이터를 랜덤 변수를 이용하여 뒤 섞는다.
        3. 파일로 존재하였던 영상데이터를 학습 데이터의 개수(5000개) 길이의 리스트로 재편성한다.
        4. png 파일에서 제공한 당초의 학습용 데이터를 knn 모델링하기 위한 학습 데이터와 테스트 데이터로 반씩 나눈다.
    5. knn 모델을 생성하고, 학습데이터로 학습시킨다. OpenCV에서는 두 함수가 필요하다. - create(), train()
    6. 나머지 반의 데이터로 테스트를 행하고 분류 성능을 분석한다.
        사용되는 함수: 결과=findNearest(테스트_데이터, k)
        k변화에 따른 정확도 변화를 관찰한다. 학습된데이터를 테스팅하니 K=1이면 100% 정확도를 보였다.



"""

# Import required packages:
import cv2
import numpy as np

# Constants:
SIZE_IMAGE_ROW = 20     # 글자 하나의 세로 해상도
SIZE_IMAGE_COL = 20     # 글자 하나의 가로 해상도
NUMBER_CLASSES = 10     # 영상 파일 안에 있는 글자의 클래스 10종(0~9)


def load_digits_and_labels(big_image):
    """Returns all the digits from the 'big' image and creates the corresponding labels for each image"""

    # (1) Load the 'big' image containing all the digits:
    # 영상은 20x20으로 구성된 문자폰트가 가로로 100개, 세로로 50개가 나열되어 있다.
    # 각 다섯줄 마다 같은 문자가 100개씩 나열되어 있다. 문자 0이 5줄, 문자 1이 5줄, ..., 문자 9가 5줄.
    # 따라서 문자 0이 500개, 문자 1이 500개, ..., 문자 9가 500개 총합 5000개의 문자 폰트가 배열되어 있다.
    digits_img = cv2.imread(big_image, 0)
    print('(1) digits_img.shape=', digits_img.shape)    # digits_img.shape= (1000, 2000)

    # (2) 입력 영상에 가로로, 세로로 몇 개의 손글씨 폰트 영상이 있는지 밝힌다.
    # 영상 파일이 가로로는 100글자, 세로로는 50 글자가 있다. 총 5,000자의 글자가 영상 파일에 있다.
    number_cols = digits_img.shape[1] / SIZE_IMAGE_COL  # 영상의 전체 가로 크기 / 폰트의 가로 크기
    print('\n(2a) number_cols=', number_cols)  # number_cols= 100.0
    number_rows = digits_img.shape[0] / SIZE_IMAGE_ROW
    print('(2b) number_rows=', number_rows)  # number_rows= 50.0

    # (3) vsplit(어레이, indices_or_sections) 함수는 전체 영상을 세로로 잘라 number_rows(50)줄로 나눈 영상을 리스트 구조체로 반환한다.
    # indices_or_sections가 숫자인 경우: 첫번째 축(row)을 따라 주어진 숫자로 나눈다.
    # indices_or_sections가 [a, b, c]인 경우 첫번재 축의 a에 나누고, b에서 나누고. c에서 나눈다.
    #  vsplit is equivalent to split with axis=0 (default),
    #  the array is always split along the first axis regardless of the array dimension.
    rows = np.vsplit(digits_img, number_rows)   # 영상을 세로로 number_rows개로 나눈다.

    print(f'\n(3) type(rows)={type(rows)} | len(rows)={len(rows)} | rows[0].shape={rows[0].shape}')
    # type(rows)=<class 'list'> | len(rows)=50 | rows[0].shape=(20, 2000)

    # (4) digits 리스트 자료에 20x20 크기의 필기체 문자 영상을 총 5000개 저장한다.
    # 마지막 단계에서는 digits 리스트를 같은 이름의 ndarray형으로 변환한다.
    digits = []
    # hsplit() 함수를 이용해 각 row(가로 축)에 대해 20 픽셀 단위로 잘라 list 구조체로 만든다. => 20 x20 화소 추출
    for row in rows:
        row_cells = np.hsplit(row, number_cols)
        for digit in row_cells:
            digits.append(digit)
    print(f'\n(4a) digits: type={type(digits)}, len={len(digits)}')
    # digits: type=<class 'list'>, len=5000

    digits = np.array(digits)
    print(f'(4b) digits: type={type(digits)}, shape={digits.shape}, len={len(digits)}')
    # digits: type=<class 'numpy.ndarray'>, shape=(5000, 20, 20), len=5000

    # Create the labels for each image:
    # >>> np.repeat(3, 4)
    # array([3, 3, 3, 3])
    # >>> x = np.array([[1,2],[3,4]])
    # >>> np.repeat(x, 2)
    # array([1, 1, 2, 2, 3, 3, 4, 4])
    print('(4c) np.arange(NUMBER_CLASSES)=', np.arange(NUMBER_CLASSES))

    # [0~9]를 0을 500번, 1을 500번, ..., 9를 500번, 총 5,000개의 레이블을 생성한다.
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    print(f'(4d) labels: type={type(labels)}, shape={labels.shape}')
    return digits, labels


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""
    # 본 정확도는 test 데이터에 대해서만 환산됨에 유의...
    accuracy = (np.squeeze(predictions) == labels).mean()   # 'mean()' returns the average of the array elements.
    return accuracy * 100   # labels가 10개라서 맞힌 갯수를 평균(mean)을 구해서 100을 곱하면 정확도가 나온다.


# =======================================================================================================
# 프로그램의 시작
# =======================================================================================================


# ------------------------------------------------------------------------------------------------------
# 1. 영상 파일을 읽어들이고 라벨링을 행한다.
# digit.png는 가로x세로=2,000x1,000의 흑백영상이다.
# 해당 영상은 문자 0~9까지의 손글씨 폰트가 5줄씩 나열되어 있다.
# 한 폰트는 20x20으로 이루어져 있으므로 한 줄마다 각 줄마다 100개의 같은 글자.
# 100개의 같은 글자 폰트가 5줄씩 나열되어 있다.
# 따라서 2,000x1,000 영상은 20x20크기의 문자가 100x50(가로x세로)개가 나열되어 있다.
# 0~9의 문자가 5줄 씩 같은 문자로 배열되어 있으므로 레이블(0~9)은 range 함수를 이용해 자체 생성한다.
# ------------------------------------------------------------------------------------------------------
print('\n@@1. trainin data(20x20 image) and labels(0~9) will be produced; ')
digits, labels = load_digits_and_labels('../data/digits.png')

# ------------------------------------------------------------------------------------------------------
# 2. 학습을 위해 영상 데이터를 랜덤 변수를 이용하여 뒤 섞는다.
# RandomState.permutation(x)
#   Randomly permute a sequence, or return a permuted range.
#   >>> np.random.permutation(10)
#   array([1, 7, 4, 3, 0, 9, 2, 5, 8, 6]) # random
#   >>> np.random.permutation([1, 4, 9, 12, 15])
#   array([15,  1,  9,  4, 12]) # random
# ------------------------------------------------------------------------------------------------------
print('\n@@2. Training data being shuffled.')
rand = np.random.RandomState(1234)      # Constructs a random number generator:
# Randomly permute the sequence:
shuffle = rand.permutation(len(digits))     # 0~4999의 수를 랜덤하게 재배열한다.
print('1) type(shuffle)=', type(shuffle), 'shuffle.shape=', shuffle.shape)
digits, labels = digits[shuffle], labels[shuffle]



# ------------------------------------------------------------------------------------------------------
# 3. 파일로 존재하였던 영상데이터를 학습 데이터의 개수(5000개) 길이의 리스트로 재편성한다.
# Compute the descriptors for all the images.
# In this case, the raw pixels are the feature descriptors
# ------------------------------------------------------------------------------------------------------
print('\n@@3. Arrange the training data as sequence list for knn-modeling\n'
      'Elements of the list are flattened images.')
# raw_descriptors: knn 모델링을 위한 학습 데이터(영상).
# 20x20의 영상을 1차원으로 나열한 영상 데이터가 5000개 있는 리스트 자료이다.
# 이것으로 ndarray로 변환해서 raw_descriptors.shape= (5000, 400)가 되게할 예정이다.
raw_descriptors = []

for img in digits:      # digits.shape= (5000, 20, 20). 5000번 loop를 수행.
    # img.flatten(): n차 배열을 1차원 배열로 만들어 반환한다.
    #   ravel()과는 달리 반환 값은 원본과 데이터를 공유하지 않는다.
    raw_descriptors.append(np.float32(img.flatten()))  # 리스트의 각 원소는 sequnce 형으로 나열된 영상 데이터이다.

print('1) raw_descriptors: type=', type(raw_descriptors), '| length=', len(raw_descriptors))
# 1) type(raw_descriptors)= <class 'list'> | len(raw_descriptors)= 5000

print('2) raw_descriptors[0].shape=', raw_descriptors[0].shape)
# 2) raw_descriptors[0].shape= (400,)

#raw_descriptors = np.squeeze(raw_descriptors)  # 한축 이 1의 원소를 갖는 자료가 없어 squeeze할 것이 없다.
# 스퀴즈 함수의 용법보다 간단한 용법으로 ndarray 타입으로 만들었다.
raw_descriptors = np.array(raw_descriptors)
print('3) raw_descriptors: type=', type(raw_descriptors), '| shape=', raw_descriptors.shape)
# 3) type(raw_descriptors)= <class 'numpy.ndarray'> | raw_descriptors.shape= (5000, 400)


# ------------------------------------------------------------------------------------------------------
# 4. png 파일에서 제공한 당초의 학습용 데이터를 knn 모델링하기 위한 학습 데이터와 테스트 데이터로 반씩 나눈다.
# ------------------------------------------------------------------------------------------------------
print('\n@@4. Spliting the data into training and testing (50% for each one)')
partition = int(0.5 * len(raw_descriptors))     # 중간 부분을 지정하는 index 번호


# numpy.split(ary, indices_or_sections, axis=0) 함수는 전체 영상을 세로로 잘라 number_rows(50)줄로 나눈 영상을 리스트 구조체로 반환한다.
# indices_or_sections가 숫자인 경우: 첫번째 축(row)을 따라 주어진 숫자로 나눈다. 2이면 전체를 2로 나눈다.
# indices_or_sections가 [a, b, c]인 경우 첫번재 축의 a에 나누고, b에서 나누고. c에서 나눈다. [2]이면 2번 index에서 나눈다.
# axis는 나누는데 기준이 되는 축으로 첫번째=0, 두번째=1, ..

# raw_descriptors를 index 번호=partition 나누고자 한다.
# raw_descriptors는 5000x400으로 구성된 ndarray이다.
# 0~2499까지는 학습용 데이터로, 2500~4999까지는 test 데이터로 활용한다.
raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])
#raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, 2)
#labels_train, labels_test = np.split(labels, 2)    # 2개로 나누어도 됨.


print('1) labels_train: type=', type(labels_train), '| shape=', labels_train.shape)
print('2) labels_test: type=', type(labels_test), '| shape=', labels_test.shape)
# 1) labels_train: type= <class 'numpy.ndarray'> | shape= (2500,)
# 2) labels_test: type= <class 'numpy.ndarray'> | shape= (2500,)


# ------------------------------------------------------------------------------------------------------
# 5. knn 모델을 생성하고, 학습데이터로 학습시킨다. OpenCV에서는 두 함수가 필요하다. - create(), train()
# 대략적인 내용은 위키피디아에서 "k-최근접 이웃 알고리즘"을 참조.
#
# 1) Creates the empty model. => retval = cv.ml.KNearest_create()
# The static method creates empty KNearest classifier. It should be then trained using StatModel::train method.
# 링크: https://docs.opencv.org/4.2.0/dd/de1/classcv_1_1ml_1_1KNearest.html#ab0ce3fa7a242f3418938592f4d935b81

# 2) Trains the statistical model. => retval = cv.ml_StatModel.train(samples, layout, responses)
# 링크: https://docs.opencv.org/4.2.0/db/d7d/classcv_1_1ml_1_1StatModel.html#aeb25a75f438864fb25af182fb4b1b96f
#   samples: training samples
#   responses: vector of responses associated with the training samples.
#   cv2.ml.ROW_SAMPLE: each training sample is a row of samples
# ------------------------------------------------------------------------------------------------------
print('\n@@5. Training KNN model - raw pixels as features')
knn = cv2.ml.KNearest_create()
knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)


# ------------------------------------------------------------------------------------------------------
# 6. 나머지 반의 데이터로 테스트를 행하고 분류 성능을 분석한다.
#   사용되는 함수: 결과=findNearest(테스트_데이터, k)
#   k변화에 따른 정확도 변화를 관찰한다. 학습된데이터를 테스팅하니 K=1이면 100% 정확도를 보였다.
#
# ret, result, neighbours, dist = findNearest(raw_descriptors_test, k)
#
#   result: 테스트 샘플 갯수(T)만큼 길이의 Tx1 ndarray. 내부 요소는 부동소수로 분류값이나, 리그레션 결과를 담고 있다.
#       Vector with results of prediction (regression or classification) for each input sample.
#       It is a single-precision floating-point vector with <number_of_samples> elements.
#   neighbours: 분류 과정에서 주변의 k개가 어떤 레벨을 가리키고 있는지 나타낸다.
#       output values for corresponding neighbors.
#       It is a single- precision floating-point matrix of <number_of_samples> * k size.
#   dist: 주변의 k개가 현 test 데이터와 얼마만큼의 떨어져 있는지 보인다.
#       output distances from the input vectors to the corresponding neighbors.
#       It is a single-precision floating-point matrix of <number_of_samples> * k size.
# ------------------------------------------------------------------------------------------------------
print('\n@@6. Test the created model..')

k = 1
ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
print('1) type(ret)=', type(ret), 'ret=', ret)     # 함수의 성공 수행 여부만 반환함.

print('2) result.shape=', result.shape, '| result[0]=', result[0])
# (2500, 1) | result[0]= [1.] 0번째 결과는 문자 1이라고 말한다.

print('3) neighbours.shape=', neighbours.shape,'| neighbours[0]=', neighbours[0] )
# (2500, 3) | neighbours[0]= [1. 1. 1.] 주변의 k=3개가 모두 문자 1로 둘러싸여 있다.

print('4) dist.shape=', dist.shape, '| dist[0]=', dist[0])
# (2500, 3) | dist[0]= [ 88687. 116511. 144120.] 주변 k=3개와 test 점과의 개별 거리.

acc = get_accuracy(result, labels_test)     # png 파일을 받을 때 부터 정답이 labels_test임을 알고 있다.
print(f"k={k}: Accuracy for test data: {acc:#6.2f}")

# 1) 학습된 데이터와 다른 test 데이터를 적용해서 KNN으로 가장 가까운 라벨을 추론했을 경우의 정확도
for k in [3, 5, 7]:    # 동률을 이루지 않도록 k는 홀수로 선정한다.
    ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
    # Compute the accuracy:
    acc = get_accuracy(result, labels_test)     # test 데이터를 분류했을 때의 정확도를 확인
    print(f"k={k}: Accuracy for test data: {acc:#6.2f}")


print()
# 2) 학습된 데이터와 같은 학습 데이터를 적용해서 KNN으로 가장 가까운 라벨을 추론했을 경우의 정확도
for k in [1, 3, 5, 7]:    # 동률을 이루지 않도록 k는 홀수로 선정한다.
    ret, result, neighbours, dist = knn.findNearest(raw_descriptors_train, k)
    # Compute the accuracy:
    acc = get_accuracy(result, labels_train)  # 학습데이터를 분류했을 때의 정확도를 확인
    print(f"k={k}: Accuracy for train data: {acc:#6.2f}")



# ---------------------------------------------------------------------------------------
# 아래는 프로그램 분석을 위한 별도로 동작하는 코딩 연습용 프로그램입니다.
# 필요시 활용하세요...
# ---------------------------------------------------------------------------------------


"""
# shuffle 연습을 위한 코딩입니다.
# 정상적인 진행을 위해서는 주석문으로 잠가두거나 지워버리세요...
# 본문 코딩에서 학습데이터를 랜덤하게 섞기 위해서 사용합니다.
# 
# RandomState.permutation(x)
#   Randomly permute a sequence, or return a permuted range.
a = np.random.permutation(10)   # 0~9까지의 수를 랜덤하게 배열.
print(a)    # [9 3 5 2 6 0 8 4 7 1]

b = np.random.permutation(a)    # 어레이를 랜덤하게 배열할 수도 있다.
print(b)    # [2 1 4 9 3 5 7 6 8 0]

c = np.random.permutation(a)    # 어레이를 랜덤하게 배열할 수도 있다.
print(c)    # [1 6 3 0 7 5 9 2 8 4]

a = np.array(range(10))     #
print(f'a: type(a)={type(a)}, a={a}')
#a: type(a)=<class 'numpy.ndarray'>, a=[0 1 2 3 4 5 6 7 8 9]

rand = np.random.RandomState(3289)  # random generator 생성
shf = rand.permutation(len(a))     # 0~9의 수를 랜덤하게 재배열한다.
print('shf: ', type(shf), shf)
# shf:  <class 'numpy.ndarray'> [9 5 2 6 1 4 8 0 7 3]

a = a[shf]
print('shuffle result, a: ', a)
# shuffle result, a:  [9 5 2 6 1 4 8 0 7 3]
exit()
"""






"""
# --- 어떤 차원의 원소가 1이 아니면 차원을 더이상 줄일 것이 없다.
# 본문에서 squeeze()를 사용하였지만 사실 squeeze() 함수 필요없음. 
# numpy.squeeze()
#   Remove single-dimensional entries from the shape of an array.
# 예1)  x = np.array([[[0], [1], [2]]])
#       print(x.shape)
#       (1, 3, 1)
#       print(np.squeeze(x).shape)
#       (3,)
# 예2)  x = ((1,2,3), (4,5,6))       <= 3x2 더 이상 줄일 차원이 없다.
#       y = np.squeeze(x)
#       print(y)
#       [[1 2 3]
#       [4 5 6]]
# 예3)
# a = np.array([[[1]]])
# print(a.shape)  # (1, 1, 1)
# b = np.squeeze(a)
# print(b.shape, b)   # () 1
# exit()

"""