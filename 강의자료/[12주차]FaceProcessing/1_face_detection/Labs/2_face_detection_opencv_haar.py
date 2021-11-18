"""
Face detection using haar feature-based cascade classifiers
    Haar cacade feature detector를 지원하는 OpenCV의 함수를 기반으로 주어진 사진에서 얼굴을 검출한다.
    정확도는 좀 떨어지지만 DNN 기반의 처리 방식보다 속도가 빠른 장점이 있다.

OpenCV provides four cascade classifiers to use for (frontal) face detection:
    1) haarcascade_frontalface_alt.xml (FA1): 22 stages and 20 x 20 haar features
    2) haarcascade_frontalface_alt2.xml (FA2): 20 stages and 20 x 20 haar features
    3) haarcascade_frontalface_alt_tree.xml (FAT): 47 stages and 20 x 20 haar features
    4) haarcascade_frontalface_default.xml (FD): 25 stages and 24 x 24 haar features


Haar cascade를 이용한 얼굴 검출 방법을 2가지 보인다.
    1) 일반화된 detectMultiScale()를 이용하는 방법
        haar classifier를 생성하고, 이것을 메소드 detectMultiScale()를 호출하여 추출한다. 모노영상으로 입력
    2) haar 전용의 함수 getFacesHAAR()를 이용하는 방법
        haar classifier를 지정하면서 getFacesHAAR() 함수를 호출한다. 칼라영상으로 입력


호환성 문제 기록
    1. 2020년 초 이 프로그램은 python 3.7.6 + opencv-contribution-python 4.0.1.24에서 작동
    2. python 3.9.2 + opencv-python 4.5.2.54 에서 안됨. 최신 버전에서 안된다고 생각했으나,
    3. 바로 실험해 본 python 3.9.2 + opencv-contribution-python 4.5.2.54 에서 잘됨. contribution 모듈이 필요한 것이었음


"""

path = '../../data/'

file = 'best-celebrity-selfie-ellen.jpg';  magnifier = 1
#file = 'ffhq-examples.png'; magnifier = 6
#file = 'StyleGAN_examples.png'; magnifier = 0.5      # 영상 크기 줄이는 것이 좋겠음.
#file = 'group-happy-friends-taking-photo_1139-267.jpg'; magnifier = 1
#file = 'olderteens2.jpg'; magnifier = 1
#file = 'bts-awkward-family-photos-2-704.jpg'; magnifier = 1
#file = 'c_370384-l_1-k_imagepuff.jpeg'; magnifier = 1
#file = 'group-photo-july-20181.jpg';    magnifier = 1
#file = '002-18.jpg'; magnifier = 2
#file = '003-17b.png'; magnifier = 2
file = 'gaffney-group.jpg';  magnifier = 1

#file = 'olderteens2.jpg'


#file = "test_face_detection.jpg"
#file = 'bts1.jpg'
#file = 'bts2.jpg'
#file = 'twice1.jpg'
#file = 'twice2.jpg'
#file = 'faces_cartoon.jpg'  # DNN보다 검출 얼굴 수가 더 많음



# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


# Load image and convert to grayscale:
fullName = path + file
img = cv2.imread(fullName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# 방법 1) 캐스케이드 클래시파이어 객체를 선언하고 이것의 메소드, detectMultiScale()를 이용한 방법
#   (주의: detectMultiScale() 메소드는 gary 영상을 필요로 한다.)
#   사례를 보이기 위해 4가지 중에서 2개만 적용하는 방법을 보인다.
#
cas_alt2 = cv2.CascadeClassifier(path + "cascades/haarcascades/haarcascade_frontalface_alt2.xml")
cas_default = cv2.CascadeClassifier(path + "cascades/haarcascades/haarcascade_frontalface_default.xml")
print(type(cas_alt2), type(cas_default))    # <class 'cv2.CascadeClassifier'> <class 'cv2.CascadeClassifier'>

# Detect faces:
faces_alt2 = cas_alt2.detectMultiScale(gray)
print(1, type(faces_alt2), faces_alt2.shape)    # 1 <class 'numpy.ndarray'> (2, 4). 2명 검출

faces_default = cas_default.detectMultiScale(gray)
print(2, type(faces_default), faces_default.shape)  # 2 <class 'numpy.ndarray'> (4, 4). 4명 검출

# 방법 2) 캐스케이드 클래시파이어를 지정하고 함수 getFacesHAAR를 이용한 방법
# 이 방법은 squeeze() 함수를 통해야 detectMultiScale() 메소드와 같은 결과를 얻는다.
# getFacesHAAR() 함수는 컬러 영상과 캐스케이드 클래시파이어를 입력 받는다.
retval, faces_haar_alt2 = cv2.face.getFacesHAAR(img, path + "cascades/haarcascades/haarcascade_frontalface_alt2.xml")
print(3.1, type(faces_haar_alt2), faces_haar_alt2.shape)  # 3.1 <class 'numpy.ndarray'> (3, 1, 4)
faces_haar_alt2 = np.squeeze(faces_haar_alt2)
print(3.2, type(faces_haar_alt2), faces_haar_alt2.shape)  # 3.2 <class 'numpy.ndarray'> (3, 4)

retval, faces_haar_default = cv2.face.getFacesHAAR(img, path + "cascades/haarcascades/haarcascade_frontalface_default.xml")
print(4.1, type(faces_haar_default), faces_haar_default.shape)  # 4.1 <class 'numpy.ndarray'> (8, 1, 4)
faces_haar_default = np.squeeze(faces_haar_default)
print(4.2, type(faces_haar_default), faces_haar_default.shape) # 4.2 <class 'numpy.ndarray'> (8, 4)

# Draw face detections:
img_faces_alt2 = show_detection(img.copy(), faces_alt2)
img_faces_default = show_detection(img.copy(), faces_default)
img_faces_haar_alt2 = show_detection(img.copy(), faces_haar_alt2)
img_faces_haar_default = show_detection(img.copy(), faces_haar_default)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 8))
plt.suptitle("Face detection using haar feature-based cascade classifiers", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_alt2, "detectMultiScale(frontalface_alt2): " + str(len(faces_alt2)), 1)
show_img_with_matplotlib(img_faces_default, "detectMultiScale(frontalface_default): " + str(len(faces_default)), 2)
show_img_with_matplotlib(img_faces_haar_alt2, "getFacesHAAR(frontalface_alt2): " + str(len(faces_haar_alt2)), 3)
show_img_with_matplotlib(img_faces_haar_default, "getFacesHAAR(frontalface_default): " + str(len(faces_haar_default)),
                         4)

# Show the Figure:
plt.show()
