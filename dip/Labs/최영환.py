import cv2
import dlib
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

test_img_path = 'd:/dip/images/'
test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg']

model_path = 'd:/dip/data/dlib_face_recog/'

pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

# 런닝맨 멤버
mb_han_lst = ['유재석', '지석진', '김종국', '하하', '송지효', '전소민', '양세찬']
mb_eng_lst = ['Yoo', 'ji', 'kim', 'ha', 'song', 'jeon', 'yang']
md_dic = {'Yoo':'유재석', 'ji':'지석진', 'kim':'김종국', 'ha':'하하',
          'song':'송지효', 'jeon':'전소민', 'yang':'양세찬'}

# threshold 값
threshold = 0.6
print("threshold=", threshold)

# dlib 얼굴 검출기 및 얼굴 인코더 로드
with open("d:/dip/data/encoding_and_label.bin", "rb") as file:
    with open("face_encoding_and_label.bin", "rb") as file:
        all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)

for i, (a_dsc, a_lbl) in enumerate(zip(all_member_encodings1d_lst, all_member1d_label_lst)):
    print(f"{i}: label={a_lbl}", end='')
    for k in range(10):  # 한 사진의 인코딩 값을 앞의 10개만 찍어본다.
        print(f"{a_dsc[k]:7.3f}", end=" ")
    print()
for i, (a_dsc, a_lbl) in enumerate(zip(all_member_encodings1d_lst, all_member1d_label_lst)):
    print(f"{i}: label={a_lbl}", end='')
    for k in range(10):  # 한 사진의 인코딩 값을 앞의 10개만 찍어본다.
        print(f"{a_dsc[k]:7.3f}", end=" ")
    print()