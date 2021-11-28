"""
Handwritten digits recognition using KNN and raw pixels as features and varying both k and the number of
training/testing images

프로그램의 의도하는 것.
    학습 데이터의 양을 늘리면 정확도(인식률)도 높아진다.
    학습 테이터를 전체 5,000개 중 10%, 20%, ...90%로 변경하였을 때의 정확도를 출력하고 그림으로 보인다.

참고 - knn_handwritten_digits_recognition_introduction.py에 쓰였던 digit.png 영상 데이터를 실험하였다.

미션 - 다음 2개의 미션을 달성하는 프로그램을 작성하시오. 제시하지 않은 조건은 그대로 적용하는 것으로 한다.
    1. train data에 대해 인식률을 확인하시오.
    2. 본 예제는 정확도를 test data에 대해서만 적용하였다.
       정확도를 test/(test+train)으로 변경하여 적용한 결과를 출력하시오.

"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10


def load_digits_and_labels(big_image):
    """Returns all the digits from the 'big' image and creates the corresponding labels for each image"""

    # Load the 'big' image containing all the digits:
    digits_img = cv2.imread(big_image, 0)

    # Get all the digit images from the 'big' image:
    number_rows = digits_img.shape[1] / SIZE_IMAGE
    rows = np.vsplit(digits_img, digits_img.shape[0] / SIZE_IMAGE)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_rows)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    # Create the labels for each image:
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels


def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""
    # 본 정확도는 test 데이터에 대해서만 환산됨에 유의...
    accuracy = (np.squeeze(predictions) == labels).mean()
    return accuracy * 100


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()





# Load all the digits and the corresponding labels:
digits, labels = load_digits_and_labels('../data/digits.png')

# Shuffle data
# Constructs a random number generator:
rand = np.random.RandomState(1234)
# Randomly permute the sequence:
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

# Compute the descriptors for all the images.
# In this case, the raw pixels are the feature descriptors
raw_descriptors = []
for img in digits:
    raw_descriptors.append(np.float32(raw_pixels(img)))
raw_descriptors = np.squeeze(raw_descriptors)

# Split data into training/testing:
split_values = np.arange(0.1, 1, 0.1)

# Create a dictionary to store the accuracy when testing:
results = defaultdict(list)
# results: 10%~90% 학습 데이터를 사용할 때의 인식율(정확도)을 저장할 공간.
# 사전자료형으로 key=학습에 사용한 데이터의 퍼센트. value=그때의 정확도


# Create KNN:
knn = cv2.ml.KNearest_create()

for split_value in split_values:    # [0.1, ..., 0.9]
    # Split the data into training and testing:
    partition = int(split_value * len(raw_descriptors))
    raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])    # partition 번째에서 리스트를 분할한다.
    labels_train, labels_test = np.split(labels, [partition])

    # Train KNN model
    print(f'\n\nTraining KNN model using {partition} training data, which is {split_value*100:#.2f}% ...')
    knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

    print(f'{len(labels_test)}: k:Acc => ', end='')
    # Store the accuracy when testing:
    for k in np.arange(1, 10):
        ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
        acc = get_accuracy(result, labels_test)
        #print(" {}".format("%.2f" % acc))
        print(f"{k}:{acc:#.1f}", end=' ')
        results[int(split_value * 100)].append(acc)

# Show all results using matplotlib capabilities:
# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("k-NN handwritten digits recognition", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

ax = plt.subplot(1, 1, 1)
ax.set_xlim(0, 10)
dim = np.arange(1, 10)

for key in results:
    ax.plot(dim, results[key], linestyle='--', marker='o', label=str(key) + "%")

plt.legend(loc='upper left', title="% training")
plt.title('Accuracy of the KNN model varying both k and the percentage of images to train/test')
plt.xlabel("number of k")
plt.ylabel("accuracy")
plt.show()
