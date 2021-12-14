"""
3차 과제, "사진 속에서 런닝맨 멤버찾기" 제출용 소스 작성을 위한 템플레이트 파일

1. 해야 할 일(보여 줄 것들) - 자세한 것은 사례화면 참조 바람.

    1) 기본 사항 - matplotlib 출력 화면에서 해야 할 일
        (1) 런닝맨 멤버가 포함된 사진에서 멤버를 찾아 사진에서 검출된 인물의 얼굴에 사각형으로 표시하고,
        (2) 사각형 박스의 좌측 상하단에 0번부터 얼굴 번호를 적어넣고,
        (3) 멤버라고 판단한 인물 박스 상단에는 영문 이니셜을 적어 넣는다.
        (4) 멤버가 아니면 unknown으로 명기한다.
        (5) 종합적인 정보를 아래와 같이 plt.title()에 적는다.(fontsize=20 추천)
            [파일의 이름] 검출한 얼굴의 수, 멤버라고 판단한 인물의 수
    2) 추가 가점 사항 - 추가 정보 제공
        (1) 수행창에서 다음의 추가 정보를 print()하여 제공: 추가 가점 사항
            얼굴 검출 번호 별 판단한 멤버의 이니셜(혹은 이름)과 그때의 유클리디언 거리
        (2) matplplot 창에서 검출된 인물 별로 그 사람이라고 판단하게된 얼굴을
            3순위까지 선택하여 유클리디언 거리 순으로 표시

2. 프로그램 작성 조건
    본 프로램에서 이름을 바꾸지 말아야 할 변수들 - 특히 앞부분에 명기한 1)~4)영역

3. 프로그램 설계 힌트
    1) 멤버의 사진을 하나만 사용해도 되지만, 조금 더 높은 성능을 기대한다면
    여러 장의 사진을 장만하여 이중의 어떤 것과 유클리디언 거리가 가까우면 그 사람으로 판별하는 것입니다.
    아마도 많이 준비할 수록 인식 성능이 높아질 가능성이 있습니다.
    인식이 잘 안되는 멤버가 있다면 그 멤버의 DB 사진의 수를 늘리는 것도 방법이 될 수 있습니다.
    2) KNN은 준비 시간이 불충분한 관계로 추천하지 않습니다.
    더 연구해봐야겠지만, KNN도 사실상 유클리디언과 큰 차이가 없는 군집화가 일어나기 때문에 큰 성능 차이가 없을 것으로 예상됩니다.
    그러나, 이미 준비한 내용이 있다면 성능은 특별히 나아지지 않아도, 기술적 가치가 있는 것으로 평가하겠습니다.


* 이 프로그램의 진행을 위해서는 다음의 파일들이 필요하다.
    1) 런닝맨 7인의 얼굴 DB: 멤버의 이니셜 폴더에 여러장의 사진을 저장해 두고 있다. 디렉토리별 10장
    2) 각 사진에 대한 128차원 디스크립터 생성이 끝나면 이를 라벨과 함께
       'encoding_and_label.bin' 파일
    3) 일단 파일이 저장되면 소스 프로그램의 2~4단계는 주석문으로 처리하고, 이 파일을 판단하는데 사용할 수 있다.

"""

import cv2
import dlib
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

# 필수 사항 -----------------------------
# 아래의 변수이름과 표현식은 바꾸지 말고 소스 그대로 상단에 노출되게 해 주십시오.

# 1) 테스트할 영상 파일이 있는 드라이브와 폴더. 실제 평가에서는 바꾸어 사용할 수 있습니다.
test_img_path = 'd:/dip/images/'

# 2) 그 폴더 안에 있는 테스트 대상 파일 이름
# 이 파일은 런닝맨 맴버와 아닌 사람들이 섞여 있습니다.
# 실제 평가에서는 바꾸어 사용할 수 있습니다. 테스트 파일이 10여개가 될 수도 있습니다.
test_file_list = ['t1.jpg', 't2.jpg', 't3.jpg', 't4.jpg', 't5.jpg']

# 3) path for dlib model: 위치 바꾸지 말고 제출해 주세요.
# 평가자의 PC에 설치해 놓고 테스트 할 것입니다.
model_path = 'd:/dip/data/dlib_face_recog/'

# 4) shape predictor와 사용하는 recognition_model
# 아래 정의된 것을 그대로 사용해 주세요.
pose_predictor_5_point = dlib.shape_predictor(model_path + "shape_predictor_5_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(model_path + "dlib_face_recognition_resnet_model_v1.dat")


# 권장사항 ------------------------------------------------------
# 꼭 자킬 필요는 없고 여러분이 선택한 방법으로 성능과 편의성을 추구하면 됩니다.

# 권장사항 1
# 멤버들의 디스크립터를 따로 저장해서 사용해야 합니다.
# 이를 위해서는 pickle 파일을 사용할 것을 권합니다.
# 인터넷에서 쉽게 찾아볼 수 있긴 하지만, 별도로 예제 프로그램도 함께 공개하겠습니다.
# 다른 이유 때문에 수행을 위해 필요한 파일이 있다면 함께 제출해 주세요.
# 그 파일은 소스 파일과 같은 위치에 있어야 하며, 별도의 설치를 요구하지 말아야 합니다.
#
# 제출하기 전에 아래와 같이 미리 인코딩된 디스크립터나 라벨 리스트를 특정 파일로 저장하는 프로그램을
# 수행해서 파일로 저장해 둡니다.
#with open("encoding_and_label.bin", "wb") as file:
#    pickle.dump((all_member_encodings1d_lst, all_member1d_label_lst), file)

# 제출할 때는 아래의 코드와 같이 저장된 파일로부터 멤버들의 디스크립터와 라벨을 모아 놓은 리스트들을 복원할 수 있습니다.
# 소스와 저장된 파일을 같이 제출하면 평가자는 쉽게 확인해 볼 수 있겠죠.
#with open("encoding_and_label.bin", "rb") as file:
#    all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)


# 권장사항 2
# 검색하고자 하는 인물은 아래와 같이 총 7인입니다.
# 아래의 변수 사용 여부는 평가에 포함되지 않습니다.
# 다만, 사각형 얼굴 검출 화면에는 아래의 영문 이름 표기를 통일해서 사용해 주십시오.
mb_han_lst = ['유재석', '지석진', '김종국', '하하', '송지효', '전소민', '양세찬']
mb_eng_lst = ['Yoo', 'ji', 'kim', 'ha', 'song', 'jeon', 'yang']
md_dic = {'Yoo':'유재석', 'ji':'지석진', 'kim':'김종국', 'ha':'하하',
          'song':'송지효', 'jeon':'전소민', 'yang':'양세찬'}

# 권장사항 3
# threshold보다 큰 유클리디언 거리는 동일 인물로 판단하지 않습니다.
# 이 값보다 작은 값중에 가장 가까운 인물로 선발하는 것을 권합니다.
# 평가에서는 이 값을 바꾸어 가면서 점검하지 않습니다.
# 제출한 대로 인식 성능을 평가할 것입니다.
# 다양한 경우에 대해 인식 성능이 고도화될 수 있도록 해야 합니다.
threshold = 0.6
print("threshold=", threshold)


# --------------------------------------------------------------------------------------------------
# 1. dlib 얼굴 검출기와 얼굴 인코더를 로드한다.
# Load 1) shape predictor, 2) face encoder and 3) face detector using dlib library
# --------------------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------------------------------
# 아래의 코드는 필수적인 것은 아닙니다.
# 프로그래밍을 위한 아이디어를 얻는 차원으로 사용하거나, 그대로 써도 상관 없습니다.
# 만약 유사한 개념을 사용한다면 피클 파일 "face_db.bin"을 꼭 같이 제출해야 합니다.
# 피클 파일을 만드는 소스 프로그램은 제출할 필요없습니다.
#
# 저장한 피클 파일로부터 모든 멤버의 얼굴 사진, 디스크립터 정보, 라벨링 리스트 자료를 읽어온다.
# ----------------------------------------------------------------------------------------------------------
with open("face_db.bin", "rb") as file:
    with open("face_encoding_and_label.bin", "rb") as file:
        all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)

for i, (a_dsc, a_lbl) in enumerate(zip(all_member_encodings1d_lst, all_member1d_label_lst)):
    print(f"{i}: label={a_lbl}", end='')
    for k in range(10):  # 한 사진의 인코딩 값을 앞의 10개만 찍어본다.
        print(f"{a_dsc[k]:7.3f}", end=" ")
    print()

# ----------------------------------------------------------------------------------------------------------
# 6. 테스트 대상 영상 파일을 읽어온다.
# 아래의 코드는 필수적인 것은 아닙니다.
# 프로그래밍을 위한 아이디어를 얻는 차원으로 사용하기 바랍니다.
# ----------------------------------------------------------------------------------------------------------

for im in test_file_list:

    matching_count = 0  # 사진 속에서 멤버 매칭이 발생한 횟수
    match_face_lst = []  # 신원이 확인된 원본 영상의 얼굴 영상, matching_count 수만큼 확보한다.

    img = cv2.imread(test_img_path + im)
    rgb = img[..., ::-1].copy()
    img2 = img.copy()       # 화면 출력용 버퍼

    # 디스플플레이 창을 생성한다.
    plt.figure(num=im)              # 필수 사항 - 신원 파악한 결과 출력용 창
    win_anlys = im + ' analysis'
    plt.figure(num=win_anlys)       # 추가 가점사항(2)를 위한 창 - 어떤 점수로 인식했는지 1~3위의 사진을 보여준다.

    print(f'\n[File: {im}] -----------------------------')

    for i, (unknown_encoding, loc) in enumerate(zip(face_dscrptr_lst, face_locations)):

        # 일단 얼굴 검출된 영역의 얼굴을 나중에 보여줄 용도로 임시로 저장한다. 추가사항 (2)의 목적
        face_cut = img[loc.top():loc.bottom(), loc.left():loc.right()].copy()

        # 검출된 얼굴을 박스로 표시한다.
        # ....생략....

        # 박스 위에 검출된 얼굴의 번호를 출력한다.
        # ....생략....

        # 검출된 얼굴의 엔코딩과 미리 준비해 놓은 인코딩을 비교하여 누군인지 판별한다.

        computed_distances_ordered, ordered_names = compare_faces_ordered \
            (all_member_encodings1d_lst, all_member1d_label_lst, unknown_encoding)

        if computed_distances_ordered[0] > threshold:
            id = 'Unknown'  # 가장 가까운 유클리디언 거리가 0.6이상이면 동일 인물로 볼 수 없다.
        else:
            # ....생략....
            match_face_lst.append(face_cut)  # 오려 놓은 얼굴을 저장한다. 글씨 없는 얼굴이어야 하는데...

        # ....생략....

        print(f"{i}: ", end='')
        if id != 'Unknown':

            # 분석화면(win_anlys)으로 넘어가 원본 영상에서 검출된 얼굴을 출력한다.
            plt.figure(num=win_anlys)
            plt.subplot(4, 8, matching_count)
            # ....생략....

            # 원본과 가장 가깝다고 판단한 3개의 후보 사진을 아래에 유클리디어 거리와 함께 열로 나열한다.
            for th in range(3):
                # ....생략....
        else:
            print("Unknown")

    # 다시 화면을 바꾸어 검출된 얼굴의 신원과 검색번호와 함께 얼굴을 사각형으로 표시한 화면을 출력한다.
    plt.figure(num=im)
    plt.imshow(img2[..., ::-1])
    plt.title(f"[File: {im}] {len(face_locations)} faces found including {matching_count} members", fontsize=20)
    plt.axis('off')

    print(f'[File: {im}] {len(face_locations)} faces found including {matching_count} members')
plt.show()
