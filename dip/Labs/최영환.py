import cv2
import dlib
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

# used methods
# =====================================================================================================================

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

# =====================================================================================================================

# test image path & test image name list
test_img_path = 'd:/dip/images/'
test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg']

# model_path
model_path = 'd:/dip/data/dlib_face_recog/'

# shape predictor & recognition_model
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

# dlib hog face detector
detector = dlib.get_frontal_face_detector()

# 런닝맨 멤버
mb_han_lst = ['유재석', '지석진', '김종국', '하하', '송지효', '전소민', '양세찬']
mb_eng_lst = ['Yoo', 'ji', 'kim', 'ha', 'song', 'jeon', 'yang']
md_dic = {'Yoo':'유재석', 'ji':'지석진', 'kim':'김종국', 'ha':'하하',
          'song':'송지효', 'jeon':'전소민', 'yang':'양세찬'}

# threshold 값
threshold = 0.6
print("threshold=", threshold)

# dlib 얼굴 검출기 및 얼굴 인코더 로드
with open(FileName, "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)