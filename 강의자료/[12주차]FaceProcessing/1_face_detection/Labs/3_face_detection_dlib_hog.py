"""
Face detection using dlib frontal face detector,
which is based on Histogram of Oriented Gradients (HOG) features
and a linear classifier in a sliding window detection approach

dlib 모듈에서 지원하는 얼굴 검출 기법은 2가지가 있다;  1) hog + SVM. 2) CNN


설치 관련 코멘트
    사전에 cmake가 설치되어 있어야 한다.
        visual studio 2017을 이용해서 필요한 모듈을 생성하는 듯. -> 확실치는 않음
    이후로는 dlib를 설치해야 한다.

"""

# Import required packages:
import cv2
import dlib

from matplotlib import pyplot as plt

path = '../../data/'
file = 'best-celebrity-selfie-ellen.jpg';  magnifier = 1
file = 'ffhq-examples.png'; magnifier = 6
#file = 'StyleGAN_examples.png'; magnifier = 0.5      # 영상 크기 줄이는 것이 좋겠음.
#file = 'group-happy-friends-taking-photo_1139-267.jpg'; magnifier = 1
#file = 'olderteens2.jpg'; magnifier = 1
#file = 'bts-awkward-family-photos-2-704.jpg'; magnifier = 1
#file = 'c_370384-l_1-k_imagepuff.jpeg'; magnifier = 1
#file = 'group-photo-july-20181.jpg';    magnifier = 1
#file = '002-18.jpg'; magnifier = 2
file = '003-17b.png'; magnifier = 2
#file = 'gaffney-group.jpg';  magnifier = 1


file = "test_face_detection.jpg"
#file = 'bts1.jpg'   # 얼굴이 사진에 비해 작아서 두번 째 옵션을 1 이상으로 선택해야 검출된다.
#file = 'bts2.jpg'
file = 'twice1.jpg'
file = 'twice2.jpg'
file = 'faces_cartoon.jpg'  # CNN보다 검출 얼굴 수가 더 많음
#file = 'faces-1.jpg'
#file = 'vikings_s.jpg'
#file = 'vikings_big.jpg'


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 5)
    return image


# Load image and convert to grayscale:
fullName = path + file
img = cv2.imread(fullName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Load frontal face detector from dlib:
detector = dlib.get_frontal_face_detector()

# Detect faces:
import time
s_time = time.time()
up_scale1 = 0    # 입력 영상 크기 그대로 사용하여 검출
up_scale1 = 1    # 입력 영상의 2배 크기로 확장하여 검출. 작은 사진일 경우 필요할 수도..

rects_1 = detector(gray, up_scale1)
e1_time = time.time() - s_time

up_scale2 = 0    # 입력 영상 크기 그대로 사용하여 검출
up_scale2 = 1    # 입력 영상의 2배 크기로 확장하여 검출. 작은 사진일 경우 필요할 수도..

rects_2 = detector(gray, up_scale2)
e2_time = time.time() - s_time
#print(f"execution time0 = {e1_time-s_time:.4f}, execution time1 = {e2_time-e1_time:.4f}")

# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 4))
plt.suptitle(f"dlib HoG face detector: file={file}, shape={img.shape}", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, f"detector(gray, {up_scale1}): num={len(rects_1)}, time={e1_time:#.4f}", 1)
show_img_with_matplotlib(img_faces_2, f"detector(gray, {up_scale2}): num={len(rects_2)}, time={e2_time:#.4f}", 2)

# Show the Figure:
plt.show()
