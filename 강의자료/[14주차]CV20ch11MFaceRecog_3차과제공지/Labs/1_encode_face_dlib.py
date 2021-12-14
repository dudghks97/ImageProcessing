"""
This script makes used of dlib library to calculate the 128-dimensional (128D) descriptor to be used for face
recognition. Face recognition model can be downloaded from:
https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2

개요
    주어진 사진 속의 인물에서 얼굴을 검출하여 해당 인물의 128 차원의 디스크립터를 화면에 출력해 보는 프로그램.

주요 사항
    위의 주 기능은 자체 개발한 face_encodings()에서 실시하고 메인에서는 이것이 반환하는 결과 다루고 있다.
    face_encodings()이 하는 일
        hog 기반의 dlib face detector를 사용하여 얼굴 검출을 시행한다.
            - 주의: 이 함수는 gray 변환된 영상을 사용한다.

"""

# Import required packages:
import cv2
import dlib
import numpy as np

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
#file = 'faces_cartoon.jpg'  # CNN보다 검출 얼굴 수가 더 많음
#file = 'faces-1.jpg'
#file = 'vikings_s.jpg'
#file = 'vikings_big.jpg'

#path = ''       # 현재 폴더에 있는 사진들 실험
#file = 'jared_2.jpg'


# 단계 1 --------------------------------------------------------------------------------------------
# shape descriptor을 구하고자하는 인물 영상을 읽어들인다.
# --------------------------------------------------------------------------------------------------

# Load image and convert to grayscale:
fullName = path + file
img = cv2.imread(fullName)
# assert condition, message: condition이 false이면 message 출력하면서 AssertionError 발생.
assert img is not None, f"img={img}: 'No image file....!"

# Convert image from BGR (OpenCV format) to RGB (dlib format):
rgb = img[:, :, ::-1]
print(f"input image shape = {rgb.shape}")


# 단계 2 --------------------------------------------------------------------------------------------
# Load 1) shape predictor, 2) face encoder and 3) face detector using dlib library
# --------------------------------------------------------------------------------------------------

# path for dlib model
model_path = '../../data/dlib_face_recog/'

# 1) shape predictor를 주어진 폴더에서 로드하여 생성한다.
# 이 객체로 얼굴에서 5개(각 눈의 양끝 4점 + 코 1)의 주요 위치를 찾는다.
# 128 디스크립터를 추출하기 위해 alignment를 할 때 필요한 것으로 추정된다.
# 2)번의 face encoder가 이 점의 위치를 받아 얼굴 영역에 대해 128차원 인코딩을 실행한다.
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
# model download: 7198 faces로 학습한 5점 랜드마크 검출 모델
# https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2

if callable(pose_predictor_5_point):
    print("\nNotice!!: 'pose_predictor_5_point' is a callable object.")

# 2) face encoder를 주어진 폴더에서 로드하여 생성한다.
# ResNet network with 29 conv layers. 7,485인, 300만개의 얼굴로 학습. 99.38% 정확도.
# model down load:
# https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

# 3) hog 기반의 dlib face detector. gray 변환된 영상을 사용한다.
detector = dlib.get_frontal_face_detector()


# 함수 정의 ----------------------------------------------------------------------------------------------
# 영상을 입력받아 내부에서 얼굴을 검출하여 이들의 128차원 face_descriptor를 ndarray로 반환해 주는 함수
# 분석의 편의를 위해 중간 중간 처리 결과를 화면에 출력하도록 하였다.
# -------------------------------------------------------------------------------------------------------
def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # hog 기반의 dlib face detector. gray 변환된 영상을 사용한다.
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    face_locations = detector(gray, number_of_times_to_upsample)
    #print(type(face_locations))     # <class 'dlib.rectangles'>

    # Detected landmarks: 한 줄로 표현할 수 있지만, 분석을 위해 여러 줄로 나누어 표현하기로 한다.
    #raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]

    raw_landmarks = []  # 5점 랜드마크 정보를 한 화면에서 검출된 수만큼 리스트로 저장하기로 한다.
    for i, f_l in enumerate(face_locations):      # 영상에서 사람의 얼굴 수 만큼 loop를 수행
        pp5 = pose_predictor_5_point(face_image, f_l)   # callable object
        raw_landmarks.append(pp5)
        #print(type(pp5))    # <class 'dlib.full_object_detection'>

    print(f'\nnumber of faces detected: len(raw_landmarks)={len(raw_landmarks)}')

    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    # 한 줄로 간략히 쓸 수 있지만 분석을 위해 여러 줄로 표현하기로 한다.
    #return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
    #        raw_landmark_set in raw_landmarks]

    face_dscrptr_lst = []
    for i, raw_landmark_set in enumerate(raw_landmarks):
        dscrptr = face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)
        dsc = np.array(dscrptr)
        print(f"face[{i}] {dsc.shape}: ", end='')
        for k in range(10):
            print(f"{dsc[k]:#7.3f}", end=" ")
        print()
        face_dscrptr_lst.append(dsc)
    return face_dscrptr_lst


# 단계 3 --------------------------------------------------------------------------------------------
# face_encodings() 함수를 호출하고, 반환받은 face descriptor 자료의 값을 출력한다.
# 반환된 값은 128개의 원소를 갖는 1차원 ndarray로 된 디스크립터들을 리스트로 묶은 자료이다.
# 검출된 얼굴 수만큼 디스크립터가 반환된다.
# --------------------------------------------------------------------------------------------------

# Calculate the encodings for every face of the image:
# 입력 영상에 대해 descriptor를 반환한다.
encodings = face_encodings(rgb, number_of_times_to_upsample=0)     # 여러 사람의 경우에는 여러 사람에 대한 자료를 반환하는 것으로 판단된다.
print(f'\n1 number of faces detected = {len(encodings)}')
print(2, f'type(encodings)={type(encodings)}')     # type(encodings)=<class 'list'>

# Show the first encoding:
print(3, f'type(encodings[0])={type(encodings[0])}')   # type(encodings[0])=<class 'numpy.ndarray'>
print(4, f'dimension of descriptor[0] = {len(encodings[0])}, shape={encodings[0].shape}')
print(5, f'encodings[0].shape={encodings[0].shape}')   # encodings[0].shape=(128,)
#print(6, f'value=\n{encodings[0]}')