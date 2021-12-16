import os
import pickle
import dlib
import cv2 as cv
import numpy as np

def face_encodings(face_image, number_of_times_to_upsample=1, num_jitters=1):
    """Returns the 128D descriptor for each face in the image"""

    # hog 기반의 dlib face detector. gray 변환된 영상을 사용한다.
    gray = cv.cvtColor(face_image, cv.COLOR_RGB2GRAY)
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


# 경로 설정
Path = 'd:/dip/data/'
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

mb_han_lst = ['유재석', '지석진', '김종국', '하하', '송지효', '전소민', '양세찬']
mb_eng_lst = ['Yoo', 'ji', 'kim', 'ha', 'song', 'jeon', 'yang']

# 멤버별 디렉토리 경로
Path_list = []
for i, name in enumerate(mb_eng_lst):
    Path_list.append(Path + 'members/' + str(i+1) + '.jpg')

# 경로 리스트 출력
print(Path_list)

img_list = []
enc_list = []
label_list = []
with open(FileName, "wb") as file:
    for i, path in enumerate(Path_list):
        img = cv.imread(path)
        assert img is not None, f"img={img}: 'No image file....!"
        rgb = img[:, :, ::-1]
        encodings = face_encodings(rgb)
        label = mb_eng_lst[i]

        # list appending
        img_list.append(rgb)
        enc_list.append(encodings)
        label_list.append(label)
    data = [img_list, enc_list, label_list]
    pickle.dump(data, file)

with open(FileName, "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)
print(all_member1d_label_lst)
print(all_member_encodings1d_lst)

