"""
This script makes used of dlib library to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics
    http://dlib.net/python/index.html

개요
    4명의 라벨링이 된 인물과 1명의 신원 불명의 사람 얼굴 사진으로 미지 인물이 4인 중 누구인지 맞추는 사례 프로그램
    5인의 얼굴 디스크립터를 구하여 미지 인물의 것이 나머지 4인의 누구와 가장 유클리디언 거리가 가까운가로 신원을 판단한다.


"""

# Import required packages:
import cv2
import dlib
import numpy as np

# path for dlib model
model_path = '../../data/dlib_face_recog/'

# Load shape predictor, face enconder and face detector using dlib library:
# model down load: https://github.com/davisking/dlib-models/blob/master/shape_predictor_5_face_landmarks.dat.bz2
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
# model down load: https://github.com/davisking/dlib-models/blob/master/dlib_face_recognition_resnet_model_v1.dat.bz2
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()     # dlib hog face detector을 사용하였음.


def compare_faces_ordered(encodings, face_names, encoding_to_check):
    """Returns the ordered distances and names
    when comparing a list of face encodings against a candidate to check"""
    # 매칭값 순으로 나열하여 반환한다. ... 작은 값부터
    # 매칭값에 따라 face_names 순서도 바꾸어 반환한다.

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def compare_faces(encodings, encoding_to_check):
    """Returns the distances when comparing a list of face encodings against a candidate to check"""

    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    # linalg.norm 링크:
    # https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # Detect faces:
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    face_locations = detector(gray, number_of_times_to_upsample)

    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]

    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# 1) 미리 누구인지 아는 사람들의 얼굴을 순서대로 기록한다.
known_image_1 = cv2.imread("jared_1.jpg")
known_image_2 = cv2.imread("jared_2.jpg")
known_image_3 = cv2.imread("jared_3.jpg")
known_image_4 = cv2.imread("obama.jpg")

# 2) 위 변수명의 순서에 따라 해당 인물의 이름을 리스트로 정의한다.
names = ["jared_1", "jared_2", "jared_3", "obama"]

# 3) 아래 미지의 인물 중 하나만 선택하고, 나머지는 주석처리한다.
# 선택된 사람이 위 4인 중에서 누구와 가장 가까운 것인가를 128차원 descriptor 정보로 결정한다.
#unknown_image = cv2.imread("jared_4.jpg")       # 미지의 영상 1
#unknown_image = cv2.imread("obama6.jpg")        # 미지의 영상 2
#unknown_image = cv2.imread("two.jpg")        # 미지의 영상 2
unknown_image = cv2.imread("obama4.jpg")        # 미지의 영상 2

# 4) Convert image from BGR (OpenCV format) to RGB (dlib format):
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# 5) Create the encodings for both known images & unknown image:
# face_encodings은 검출된 인물들의 descriptor 정보를 각 사람마다 ndarray 데이터로 만들어
# 리스트 자료형으로 반환한다.
# 여러 사람이 있는 사진일 경우를 대비해 그 중 0번째를 반환 받는다.
# 미지의 인물 사진도 0번째 인물 사진에 대해 비교할 것이다.

known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding,
                   known_image_3_encoding, known_image_4_encoding]

unknown_encoding = face_encodings(unknown_image)[0]         # 여러 사람일 때는 index 번호를 바꿀 수 있다.


# 6) Compare faces:
# compare_faces() 함수는 unknown_encoding을 known_encodings list에 있는
# 여러개의 encoding에 비교한 유클리디언 거리(각 차원별 오차를 제곱한 것을 더해서 root를 취함)를 리스트로 반환한다.
# 0.6이하면 동일 인물로 본다.
# known_encodings[0].shape=(128,), unknown_encoding.shape=(128,)
computed_distances = compare_faces(known_encodings, unknown_encoding)

# 7) Print obtained results: 이름순으로 나열...
print(f'name              =', end=' ')
for n in names:
    print(f"{n.rjust(12)}", end=' ')
print()

print(f'matching distance =', end=' ')
for v in computed_distances:
    print(f"{v:#12.7f}", end=' ')
print("\n")


# 8) Print obtained results: 매칭값 순으로 나열하여 반환한다. ... 작은 값부터
# 매칭값에 따라 names 순서도 바꾸어 반환한다.
computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)

#print(ordered_names)
print(f'name              =', end=' ')
for n in ordered_names:
    print(f"{n.rjust(12)}", end=' ')
print()

#print("computed_distances_ordered = ", computed_distances_ordered)
print(f'matching distance =', end=' ')
for v in computed_distances_ordered:
    print(f"{v:#12.7f}", end=' ')
print("\n")
print(f"The unknown person is identified as '{ordered_names[0]}'.")

