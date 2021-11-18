"""
This script makes used of dlib library to calculate the 128D descriptor to be used for face recognition
and compare the faces using some distance metrics

http://dlib.net/python/index.html
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
    """Returns the ordered distances and names when comparing a list of face encodings against a candidate to check"""

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, face_names)))


def compare_faces(encodings, encoding_to_check):
    """Returns the distances when comparing a list of face encodings against a candidate to check"""

    return list(np.linalg.norm(encodings - encoding_to_check, axis=1))


def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # Detect faces:
    face_locations = detector(face_image, number_of_times_to_upsample)
    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for
            raw_landmark_set in raw_landmarks]


# Load images:
known_image_1 = cv2.imread("jared_1.jpg")
known_image_2 = cv2.imread("jared_2.jpg")
known_image_3 = cv2.imread("jared_3.jpg")
known_image_4 = cv2.imread("obama.jpg")
#unknown_image = cv2.imread("jared_4.jpg")       # 미지의 영상 1
#unknown_image = cv2.imread("obama6.jpg")        # 미지의 영상 2
unknown_image = cv2.imread("two.jpg")        # 미지의 영상 2

# Convert image from BGR (OpenCV format) to RGB (dlib format):
known_image_1 = known_image_1[:, :, ::-1]
known_image_2 = known_image_2[:, :, ::-1]
known_image_3 = known_image_3[:, :, ::-1]
known_image_4 = known_image_4[:, :, ::-1]
unknown_image = unknown_image[:, :, ::-1]

# Crate names for each loaded image:
#names = ["jared_1.jpg", "jared_2.jpg", "jared_3.jpg", "obama.jpg"]
# label과 비슷한 성격의 자료이므로 파일이름으로 착각하지 않도록 파일 확장자를 제거한 이름을 사용하겠다.
names = ["jared_1", "jared_2", "jared_3", "obama"]

# Create the encodings:
# face_encodings은 128차원의 ndarray 데이터들을 사람 얼굴에 따라 리스트 자료형으로 반환하는데
# 그중 0번째를 반환 받는다. known 영상이 한 사람이므로...
known_image_1_encoding = face_encodings(known_image_1)[0]
known_image_2_encoding = face_encodings(known_image_2)[0]
known_image_3_encoding = face_encodings(known_image_3)[0]
known_image_4_encoding = face_encodings(known_image_4)[0]
known_encodings = [known_image_1_encoding, known_image_2_encoding,
                   known_image_3_encoding, known_image_4_encoding]
unknown_encoding = face_encodings(unknown_image)[0]         # 여러 사람일 때는 index 번호를 바꿀 수 있다.

# Compare faces:
# unknown_encoding을 known_encodings list에 비교했을 때 유클리디언 거리를 계산하여 반환한다.

computed_distances = compare_faces(known_encodings, unknown_encoding)
print(f'known_encodings[0].shape={known_encodings[0].shape}')       # known_encodings[0].shape=(128,)
print(f'unknown_encoding.shape={unknown_encoding.shape}')       # unknown_encoding.shape=(128,)

computed_distances_ordered, ordered_names = compare_faces_ordered(known_encodings, names, unknown_encoding)

# Print obtained results:
print(f'name             ={names}')
print(f'matching distance={computed_distances}\n')

print(computed_distances_ordered)
print(ordered_names)

