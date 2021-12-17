import cv2 as cv
import dlib
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

# used methods
# =====================================================================================================================

def compare_faces_ordered(faces, encodings, face_names, encoding_to_check):
    """Returns the ordered distances and names
    when comparing a list of face encodings against a candidate to check"""
    # 매칭값 순으로 나열하여 반환한다. ... 작은 값부터
    # 매칭값에 따라 face_names 순서도 바꾸어 반환한다.

    distances = list(np.linalg.norm(encodings - encoding_to_check, axis=1))
    return zip(*sorted(zip(distances, faces, face_names)))


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

# =====================================================================================================================

# test image path & test image name list
#test_img_path = 'd:/dip/images/'
test_img_path = '../images/'
test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg']
#test_file_list = ['t1.jpg']

# model_path
#model_path = 'd:/dip/data/dlib_face_recog/'
model_path = '../data/dlib_face_recog/'

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
# dlib hog face detector
detector = dlib.get_frontal_face_detector()

# face db 파일
face_db = '../data/face_db.bin'
with open(face_db, "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)
    print('successfully loaded!')

print(len(all_member_encodings1d_lst))
print(all_member1d_label_lst)

# test img load & compare
for im in test_file_list:

    matching_count = 0  # 사진 속에서 멤버 매칭이 발생한 횟수
    match_face_lst = []  # 신원이 확인된 원본 영상의 얼굴 영상, matching_count 수만큼 확보한다.

    img = cv.imread(test_img_path + im)
    rgb = img[..., ::-1].copy()
    img2 = img.copy()       # 화면 출력용 버퍼

    # 디스플레이 창을 생성한다.
    plt.figure(num=im)              # 필수 사항 - 신원 파악한 결과 출력용 창
    win_anlys = im + ' analysis'
    plt.figure(num=win_anlys)       # 추가 가점사항(2)를 위한 창 - 어떤 점수로 인식했는지 1~3위의 사진을 보여준다.

    print(f'\n[File: {im}] -----------------------------')
    # face_locations : 얼굴 사진, face_dscrptr_lst: 디스크립터 리스트
    face_locations, face_dscrptr_lst = face_encodings(rgb, number_of_times_to_upsample=0)

    for i, (unknown_encoding, loc) in enumerate(zip(face_dscrptr_lst, face_locations)):

        # 일단 얼굴 검출된 영역의 얼굴을 나중에 보여줄 용도로 임시로 저장한다. 추가사항 (2)의 목적
        face_cut = img[loc.top():loc.bottom(), loc.left():loc.right()].copy()

        # 검출된 얼굴의 엔코딩과 미리 준비해 놓은 인코딩을 비교하여 누군인지 판별한다.
        computed_distances_ordered, ordered_faces, ordered_names = compare_faces_ordered \
            (all_member_face, all_member_encodings1d_lst, all_member1d_label_lst, unknown_encoding)

        # 판별결과를 통해 검출된 얼굴에 대한 처리 진행
        if computed_distances_ordered[0] > threshold:
            id = 'Unknown'  # 가장 가까운 유클리디언 거리가 0.6이상이면 동일 인물로 볼 수 없다.
        else:
            id = ordered_names[0]           # id : 일치하는 멤버 이름
            face_cut = cv.cvtColor(face_cut, cv.COLOR_BGR2RGB)
            match_face_lst.append(face_cut)  # 오려 놓은 얼굴을 저장한다.

        # 검출된 얼굴을 박스로 표시한다.
        font = cv.FONT_HERSHEY_DUPLEX
        cv.rectangle(img2, (loc.left() + 6, loc.top() + 6), (loc.right(), loc.bottom()), (255, 0, 0), 4)

        # 박스 위에 검출된 얼굴의 번호를 출력한다.
        cv.putText(img2, str(i), (loc.left() - 30, loc.top() + 30), font, 1.5, (0, 0, 255), 3)

        # 박스 위에 검출된 얼굴의 이름을 출력한다.
        cv.putText(img2, id, (loc.left(), loc.top() - 5), font, 1.0, (0, 0, 255), 3)

        # 결과 출력 부분
        print(f"face {i}: ", end='')
        if id != 'Unknown':
            matching_count += 1
            print(f"{ordered_names[0]}={computed_distances_ordered[0]:#.3f}")

            # 분석화면(win_anlys)으로 넘어가 원본 영상에서 검출된 얼굴을 출력한다.
            plt.figure(num=win_anlys)
            plt.subplot(4, 8, matching_count)
            plt.axis('off')
            plt.imshow(face_cut)

            # 원본과 가장 가깝다고 판단한 3개의 후보 사진을 아래에 유클리디어 거리와 함께 열로 나열한다.
            for th in range(3):
                face = ordered_faces[th]
                plt.subplot(4, 8, (8 * (th + 1)) + matching_count)
                plt.title(f"{computed_distances_ordered[th]:#.3f}")
                plt.axis('off')
                plt.imshow(ordered_faces[th])
        else:
            print("Unknown")

    # 다시 화면을 바꾸어 검출된 얼굴의 신원과 검색번호와 함께 얼굴을 사각형으로 표시한 화면을 출력한다.
    plt.figure(num=im)
    plt.imshow(img2[..., ::-1])
    plt.title(f"[File: {im}] {len(face_locations)} faces found including {matching_count} members", fontsize=20)
    plt.axis('off')

    print(f'[File: {im}] {len(face_locations)} faces found including {matching_count} members')
plt.show()
