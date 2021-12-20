import os
import pickle
import dlib
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# face_encodings은 128차원의 ndarray 데이터들을 사람 얼굴에 따라 리스트 자료형과 얼굴 위치좌표를 반환
def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=2):
    """Returns the 128D descriptor for each face in the image"""
    # Detect faces:
    # hog 기반의 dlib face detector. gray 변환된 영상을 사용.
    gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
    face_locations = detector(gray, number_of_times_to_upsample)
    # Detected landmarks:
    raw_landmarks = [pose_predictor_5_point(face_image, face_location) for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return face_locations, [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
                            for raw_landmark_set in raw_landmarks]


# 경로 설정
Path = '../data/'
Name = "face_db.bin"
FileName = Path + Name

# model path
model_path = '../data/dlib_face_recog/'

# 4) shape predictor와 사용하는 recognition_model
# 아래 정의된 것을 그대로 사용해 주세요.
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")

detector = dlib.get_frontal_face_detector()

if callable(pose_predictor_5_point):
    print("\nNotice!!: 'pose_predictor_5_point' is a callable object.")

# 멤버별 디렉토리 경로
Path_list = []
dir_path = Path + 'members/'

# db 생성용 리스트
face_list = []
enc_list = []
label_list = []
with open(FileName, "wb") as file:
    for name in os.listdir(dir_path):
        path = dir_path + name
        print(path)
        img = cv.imread(path)
        rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        print('converting is over')
        # get descriptor & face locations
        faces, encodings = face_encodings(rgb)
        face_img = rgb[faces[0].top():faces[0].bottom(), faces[0].left():faces[0].right(), :]
        label = name[0:name.find('_')]

        # list appending
        face_list.append(face_img)
        enc_list.append(encodings[0])   # 0번째 원소만 추가
        label_list.append(label)
    # dump data to file
    data = [face_list, enc_list, label_list]
    pickle.dump(data, file)

with open(FileName, "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)

print(all_member1d_label_lst)
print(all_member_encodings1d_lst)

plt.figure()

for i, img in enumerate(all_member_face):
    plt.subplot(10, 7, i+1)
    plt.axis('off')
    img = np.array(img)
    plt.imshow(img)

print('=================================')
print(len(all_member_face))
print(type(all_member_face))
print(type(all_member_face[0]))
print(all_member_face[0].shape)
print('=================================')
print(len(all_member_encodings1d_lst))
print(type(all_member_encodings1d_lst))
print(type(all_member_encodings1d_lst[0]))
print(all_member_encodings1d_lst[0].shape)
print('=================================')
print(len(all_member1d_label_lst))
print(type(all_member1d_label_lst))
print(type(all_member1d_label_lst[0]))

plt.show()
exit(0)