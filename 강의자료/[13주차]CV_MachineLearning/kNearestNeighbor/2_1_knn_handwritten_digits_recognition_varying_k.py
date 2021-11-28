"""
Handwritten digits recognition using KNN and raw pixels as features and varying k
개요
    digits.png 파일에 있는 5000개의 숫자 영상을 50%는 학습 영상으로 나머지 50%는 테스트 영상으로 활용한다.
    k를 1~9까지 변화시켜 가면서 정확도를 점검해 본다.
    이때 학습과 테스트에 사용하는 데이터는 고정하였다.


"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
# 구태여 colections 모듈을 사용할 필요는 없었는데
# defaultdict 모듈을 사용함으로써 향후 10%, ..90% 까지 증가시켰을 때의 인식 성능을 점검할 목적으로 수정하기 용이하다.


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

# At this point we split the data into training and testing (50% for each one):
partition = int(0.5 * len(raw_descriptors))
raw_descriptors_train, raw_descriptors_test = np.split(raw_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])
print('type(labels_test)=', type(labels_test), ' | labels_test.shape=', labels_test.shape)

# Train the KNN model:
print('Training KNN model - raw pixels as features')
knn = cv2.ml.KNearest_create()
knn.train(raw_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)

# Create a dictionary to store the accuracy when testing:
results = defaultdict(list)
print(results['50'])    # empty list를 반환한다. 최소한 오류가 발생하지는 않는다.
# 구태여 colections 모듈을 사용할 필요는 없었는데
# 학습에 사용한 문자 폰트의 비율을 10%, ..90% 까지 증가시켰을 때의 향후 비율에 따른 인식 성능을 점검할 목적으로 활용한 듯하다.

for k in np.arange(1, 10):
    ret, result, neighbours, dist = knn.findNearest(raw_descriptors_test, k)
    acc = get_accuracy(result, labels_test)
    #print(" {}".format("%.2f" % acc))
    print(f"k={k} : {acc:#.2f}")
    results['50'].append(acc)   # 50%만 학습시켰으므로 그외의 비율은 없다.

#print(results['50'])

# Show all results using matplotlib capabilities:
# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("k-NN handwritten digits recognition", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

ax = plt.subplot(1, 1, 1)
ax.set_xlim(0, 10)
dim = np.arange(1, 10)

for key in results:
    print(type(key), key)
    ax.plot(dim, results[key], linestyle='--', marker='o', label="50%")

#print(len(results))

plt.legend(loc='upper left', title="% training")
plt.title('Accuracy of the K-NN model varying k')
plt.xlabel("number of k")
plt.ylabel("accuracy")
plt.show()
