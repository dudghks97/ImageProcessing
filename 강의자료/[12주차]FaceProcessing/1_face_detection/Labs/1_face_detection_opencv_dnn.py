"""
Face detection using OpenCV DNN face detector

OpenCV가 사용하는 DNN 모델 - 3.3부터 contrib에서 main으로 옮겨왔다.
    Single Shot MultiBox Detector (SSD) framework using a ResNet-10 network.
    Caffe, TensorFlow, Torch, and Darknet 등의 DNN frame work를 사용.

OpenCV가 지원하는 pre-training face detectors
    Face detector (FP16): Floating-point 16 version of the original Caffe implementation (5.1 MB)
    Face detector (UINT8): 8-bit quantized version using TensorFlow (2.6 MB)
    각 모델은 다음의 2개 파일이 필요
        1) Caffe model의 경우
            *.caffemodel: contains the weights for the actual layers.
            deploy.prototxt: defines the model architecture.
        2) TensorFlow model의 경우
            *.pb: contains the weights for the actual layers.
            *.pbtxt: defines the model architecture

"""
# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

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

#file = "test_face_detection.jpg"
file = 'bts1.jpg'
#file = 'bts2.jpg'
#file = 'twice1.jpg'
#file = 'twice2.jpg'
#file = 'faces_cartoon.jpg'  # readNetFromCaffe, readNetFromTensorflow 2개 모델에 대해 43명 검출로 결과가 같음. 단, %는 다름.

# Load image:
fullName = path + file
image = cv2.imread(fullName)

# 색상 평면 배열을 BGR -> RGB로 바꾸어 matplotlib에서 출력하는 함수
def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""
    #ax = plt.subplot(1, 1, pos)
    plt.imshow(color_img[:, :, ::-1])
    plt.title(title)
    plt.axis('off')


# Load pre-trained model:
# deploy.prototxt: text description of the network architecture
# res10_300x300_ssd_iter_140000_fp16.caffemodel: caffemodel file with learned network

# 1) Caffe Model의 경우
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")    # 99.922%/97.495%/79.345%
# Tensorflow Model의 경우
#net = cv2.dnn.readNetFromTensorflow("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")  # 99.912%/97.502%/79.021%


# Get dimensions of the input image (to be used later):
(h, w) = image.shape[:2]

# Create 4-dimensional blob(NCHW) from image:
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)
print(f'input imge shape={blob.shape}')

# Set the blob as input and obtain the detections:
net.setInput(blob)              # 입력을 연결해 준다.

start = time.time()
detections = net.forward()      # 네트워트를 가동한다. 입력=> 네트워크 => 출력
print(f'time[sec.] for OpenCV DNN face detect={time.time()-start:#.2f}')

print(type(detections), detections.shape)           # <class 'numpy.ndarray'> (1, 1, 200, 7)

# Initialize the number of detected faces counter detected_faces:
detected_faces = 0

# Iterate over all detections:
for i in range(0, detections.shape[2]):
    # Get the confidence (probability) of the current detection:
    confidence = detections[0, 0, i, 2] # i번째 검출 영역의 confidence

    # Only consider detections if confidence is greater than a fixed minimum confidence:
    if confidence > 0.7:
        # Increment the number of detected faces:
        detected_faces += 1
        # Get the coordinates of the current detection:
        #print(detections[0, 0, i, 3:7])     # 정규화된 검출된 위치
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the detection and the confidence:
        text = "{:.1f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
        cv2.putText(image, text, (startX, y+4), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, lineType= cv2.LINE_AA, color=(0, 0, 255), thickness=3)
        cv2.putText(image, text, (startX, y+4), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, lineType= cv2.LINE_AA, color=(255, 255, 255), thickness=1)

# Create the dimensions of the figure and set title:
#fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using OpenCV DNN face detector", fontsize=14, fontweight='bold')
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)
#fig.patch.set_facecolor('silver')

# Plot the images:

show_img_with_matplotlib(image, "DNN face detector: " + str(detected_faces), 1)
#plt.imshow(image); plt.axis('off')   # 영상이 RGB 인 것을 전제하기 때문에 올바른 색상 표현이 안됨.

# Show the Figure:
plt.show()
