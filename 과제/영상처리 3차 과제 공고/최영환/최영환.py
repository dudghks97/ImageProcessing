import cv2 as cv
import dlib
import numpy as np
import pickle
from matplotlib import pyplot as plt

# figure 창이 20개가 넘어갈 경우 발생하는 RuntimeWarning 방지를 위한 코드
plt.rcParams.update({'figure.max_open_warning': 0})

# 필수 사항 -----------------------------
# 아래의 변수이름과 표현식은 바꾸지 말고 소스 그대로 상단에 노출되게 해 주십시오.

# 1) 테스트할 영상 파일이 있는 드라이브와 폴더. 실제 평가에서는 바꾸어 사용할 수 있습니다.
test_img_path = 'd:/dip/images/'


# 2) 그 폴더 안에 있는 테스트 대상 파일 이름
# 이 파일은 런닝맨 맴버와 아닌 사람들이 섞여 있습니다.
# 실제 평가에서는 바꾸어 사용할 수 있습니다. 테스트 파일이 10여개가 될 수도 있습니다.
#test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg']
test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg', 't6.jpg', 't7.jpg', 't8.jpg', 't9.jpg', 't10.jpg',
                  't11.jpg', 't12.jpg']

# 3) path for dlib model: 위치 바꾸지 말고 제출해 주세요.
# 평가자의 PC에 설치해 놓고 테스트 할 것입니다.
model_path = 'd:/dip/data/dlib_face_recog/'

# 4) shape predictor와 사용하는 recognition_model
# 아래 정의된 것을 그대로 사용해 주세요.
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")


# methods
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
    raw_landmarks = [pose_predictor_5_point(face_image, face_location)
                     for face_location in face_locations]
    # Calculate the face encoding for every detected face using the detected landmarks for each one:
    return face_locations, \
           [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters))
                            for raw_landmark_set in raw_landmarks]

# =====================================================================================================================
# dlib 얼굴 검출기 및 얼굴 인코더 로드
# dlib hog face detector
detector = dlib.get_frontal_face_detector()

# face db 파일
face_db = 'face_db.bin'
with open(face_db, "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)
    print('successfully loaded!')

# threshold 값 지정
threshold = 0.6
threshold = 0.4
threshold = 0.45
print("threshold=", threshold)

# test img load & compare
for im in test_file_list:

    matching_count = 0  # 사진 속에서 멤버 매칭이 발생한 횟수
    match_face_lst = []  # 신원이 확인된 원본 영상의 얼굴 영상, matching_count 수만큼 확보한다.

    img = cv.imread(test_img_path + im)
    rgb = img[..., ::-1].copy()
    img2 = img.copy()       # 화면 출력용 버퍼

    # 디스플레이 창을 생성한다.
    plt.figure(num=im)              # 신원 파악한 결과 출력용 창
    win_anlys = im + ' analysis'
    plt.figure(num=win_anlys)       # 어떤 점수로 인식했는지 1~3위의 사진 출력용 창

    print(f'\n[File: {im}] -----------------------------')

    # 입력된 사진에 대한 인코딩 진행
    face_locations, face_dscrptr_lst = face_encodings(rgb,
                                                      number_of_times_to_upsample=0)

    # 검출된 모든 얼굴에 대한 처리 진행
    for i, (unknown_encoding, loc) in enumerate(zip(face_dscrptr_lst, face_locations)):
        # 검출된 얼굴을 나중에 보여줄 용도로 임시로 저장한다.
        face_cut = img[loc.top():loc.bottom(), loc.left():loc.right()].copy()
        face_cut = face_cut[..., ::-1]  # RGB 형식으로 변환

        # 검출된 얼굴의 인코딩과 미리 준비해 놓은 인코딩을 비교하여 누군인지 판별한다.
        computed_distances_ordered, ordered_faces, ordered_names = compare_faces_ordered \
            (all_member_face, all_member_encodings1d_lst, all_member1d_label_lst, unknown_encoding)

        # 판별결과를 통해 검출된 얼굴에 대한 처리 진행
        if computed_distances_ordered[0] > threshold:
            id = 'Unknown'
        else:
            id = ordered_names[0]               # id : 일치하는 멤버 이름

        # 검출된 얼굴을 박스로 표시한다.
        font = cv.FONT_HERSHEY_DUPLEX
        cv.rectangle(img2, (loc.left() + 6, loc.top() + 6), (loc.right(), loc.bottom()), (255, 0, 0), 4)
        # 박스 위에 검출된 얼굴의 번호를 출력한다.
        cv.putText(img2, str(i), (loc.left() - 30, loc.top() + 30), font, 1.5, (0, 0, 255), 3)
        # 박스 위에 검출된 얼굴의 이름을 출력한다.
        cv.putText(img2, id, (loc.left(), loc.top() - 5), font, 1.0, (0, 0, 255), 2)

        # 결과 출력 부분
        print(f"face {i}: ", end='')
        # 런닝맨 멤버 중 한명인 경우
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
        # 런닝맨 멤버가 아닌 경우
        else:
            print("Unknown")

    # 다시 화면을 바꾸어 검출된 얼굴의 신원과 검색번호와 함께 얼굴을 사각형으로 표시한 화면을 출력한다.
    plt.figure(num=im)
    plt.imshow(img2[..., ::-1])
    plt.title(f"[File: {im}] {len(face_locations)} faces found including {matching_count} members", fontsize=20)
    plt.axis('off')

    print(f'[File: {im}] {len(face_locations)} faces found including {matching_count} members')
plt.show()
exit(0)
