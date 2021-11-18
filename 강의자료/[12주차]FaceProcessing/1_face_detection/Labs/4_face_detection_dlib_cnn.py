"""
Face detection using dlib CNN face detector using a pre-trained model (712 KB)
from http://dlib.net/files/mmod_human_face_detector.dat.bz2.
- 단점: 해상도가 적정한 수준이어야 한다. 크기가 다른 얼굴 인식에 취약하다.
        dlib hog 보다 훨씬 정확하지만, CPU에서 수행하면 속도가 매우 느리다.
        => 이를 해소하기 위해서는 GPU를 사용해야 하며, dlib를 GPU 용으로 다시 컴파일해야 한다.
        => 그러나, 최근 실험의 결과에 따르면 dlib를 다시 컴파일 안했지만 고속으로 처리되는 것을 확인하였다.
           설치했었던 사례: cmake 모듈을 먼저 설치한다. 이후 dlib를 설치한다.
           추정: 아마도 GPU으로도 컴파일링된 모듈이 설치된 것으로 보인다. 다운받아서 설치되었는지, 실시간 컴파일 했는지는 확실치 않다.
           GPU 실행은 자동 감지로 이루어지는 것 같다.


"""
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
file = 'olderteens2.jpg'

#file = "test_face_detection.jpg"     # 크기 1/2로 줄여도 잘됨.
#file = 'bts1.jpg'   # 얼굴이 사진에 비해 작아서 두번 째 옵션을 1, 2로 선택해야 검출된다.
#file = "bts2.jpg"    # bts1은 사진이 작아 인식이 안됨. bts2는 크기 줄이면 안됨.
#file = 'bts2.jpg'
#file = "twice1.jpg"  # 줄이면 안됨. 4명만 인식
#file = "twice2.jpg"      # 줄이면 안됨. 4명만 인식 448x600
#file = "twice3.jpg"      # 줄이면 안됨. 5명만 인식
file = 'faces_cartoon.jpg'  # DNN보다 검출 얼굴 수가 더 많음
#file = 'faces-1.jpg'
#file = 'vikings_s.jpg'
#file = 'vikings_big.jpg'

fullName = path + file

# Import required packages:
import cv2
import dlib
from matplotlib import pyplot as plt
import time

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    # faces contains a list of mmod_rectangle objects
    # The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score
    # Therefore, we iterate over the detected mmod_rectangle objects accessing dlib.rect to draw the rectangle

    for face in faces:  # 얼굴 영역에 4각형 그리는 함수
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()),
                      (255, 0, 0), 3)   # 파란색, 선의 두께
    return image

# Load image and convert to grayscale:
img = cv2.imread(fullName)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print('original img.shape=', img.shape)

# Load CNN detector from dlib:
cnn_face_detector = dlib.cnn_face_detection_model_v1(path+"dlib_face_detect/mmod_human_face_detector.dat")

# Detect faces:
start = time.time()
upsample_time = 0   # 입력영상 upsampling 안한다.
upsample_time = 1   # up sampling을 1회 더 한 영상에 대해서 검출을 시도한다. 작은 얼굴을 찾아낼 수 있다.
#upsample_time = 3   # up sampling을 실시할 수록 시간은 더 늘어간다. 메모리 용량 초과를 불어올 수 있다.
rects = cnn_face_detector(img, upsample_time)
end = time.time()
print(f'time[sec.] for face detect= {end-start:#.4f}')

# Draw face detections:
img_faces = show_detection(img.copy(), rects)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 6))
plt.suptitle(f"dlib CNN face detector\nfile={file}, shape={img.shape}", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces, f"cnn_face_detector(img, {upsample_time}): num={len(rects)}," \
    + f" time={end-start:#.4f}", 1)

# Show the Figure:
plt.show()




"""

# 용도 폐기 - 향후 개선을 위해 남겨 둠
# 사진과 검출가능한 얼굴의 크기에 따라 사진 크기를 조절해서, 
# upsampling 안하고, 문제를 해결해 보랴고 했으나, 아직 확정지을 수준이 아님.
# Resize the image to attain reasonable speed:
#img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)   # 1) 특정 비율로 줄이기

desired_width = 600      # 희망하는 가로 사이즈      # 2) 특정 크기로 줄이기
img = cv2.resize(img, (500, int(img.shape[0] * 500/img.shape[1])), fx=0.5, fy=0.5)
print('adjusted img.shape=', img.shape)


"""
