"""
Detecting facial landmarks using face_recognition module

face_recognition module의 함수를 이용하여 얼굴의 랜드마크를 검출한다.
    1. face_recognition module은 내부적으로 dlib의 HOG, CNN detector 사용
    2. 실험에 의하면 pyCharm의 Setting에서는 설치할 수 없다.
        1) pip로 설치 가능: pip install face_recognition
        먼저 cmake가 설치되어 있어야 한다.
        내 경우는 pip 명령으로 설치하였음, 어쩌면 visual studio가 필요할 지 모른다(확실치 않음).
        2) 수동으로 설치하는 방법(C++ 컴파일러가 있어야 한다): https://thecodingnote.tistory.com/8


미션
    얼굴 영역을 먼저 검출하고, 랜드마크를 구하지는 않았다.
        만약 얼굴을 먼저 검출하고 랜드마크를 검출한다면 속도 향상이 이루어 질 것으로 기대한다.
        어쩌면 모듈 자체가 얼굴 영역 검출을 내부에서 먼저 시행할 수도 있다.
        실험으로 확인해 볼 필요가 있다.

함수 소개
    https://face-recognition.readthedocs.io/en/latest/face_recognition.html#module-face_recognition.api
    face_recognition.face_landmarks(face_image, face_locations=None, model='large')
        face_image – image to search
        face_locations – Optionally provide a list of face locations to check.
        model – Optional - which model to use. “large” (default)
            or “small” which only returns 5 points but is faster.

검출되는 랜드마크 정보
    1) model='large'일 경우(default): 9 landmarks
        ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    2) model='small'일 경우: 3 landmarks
        ['nose_tip', 'left_eye', 'right_eye']


링크
    함수 사용 문서
        https://face-recognition.readthedocs.io/en/latest/readme.html



"""

# Import required packages:
import cv2
import face_recognition
from matplotlib import pyplot as plt
import numpy as np

path = '../../data/'
file = "test_face_detection.jpg"
file = 'bts1.jpg'
file = 'bts2.jpg'
#file = 'twice1.jpg'
file = 'twice2.jpg'
#file = 'faces_cartoon.jpg'  # readNetFromCaffe, readNetFromTensorflow 2개 모델에 대해 43명 검출로 결과가 같음. 단, %는 다름.
#file = 'Gogh.jpg'          # 고흐 초상화
file = 'face_test.png'      # 한 사람만 존재
#file = "Lee_DiCaprio.JPG"  # 두 사람만 존재

# Load image:
fullName = path + file
image = cv2.imread(fullName)
assert image is not None, f'\nfile not found: {fullName}'

def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# ======================================================================================================================
# Lab 1 : 얼굴의 주요 랜드마크 9개, 3개를 점으로 마킹하여 출력한다.
# ======================================================================================================================

# Create images to show the results:
image_large = image.copy()
image_small = image.copy()

# Convert the image from BGR color (which OpenCV uses) to RGB color:
rgb = image[:, :, ::-1]

# ---------------------------------------------------------------------------------------------
# 단계 1: large model로 검출한다.
# 얼굴의  주요 랜드마크 9개를 점으로 마킹한다.
# face_recognition.face_landmarks(face_image, face_locations=None, model='large')
# large: 9개의 key(landmark)가 검출. small: 3개(양눈, 코)의 key가 검출
# ---------------------------------------------------------------------------------------------

# Detect 68 landmarks:
# 랜드마크 검출 결과가 list 형태로 반환된다.
face_landmarks_list_large = face_recognition.face_landmarks(rgb)   # model to use=“large” (default)


# Print detected landmarks: 자료형과 그 내부 값을 출력해 본다.
def print_landmark_info(face_landmarks_list):
    print(f'1) type(face_landmarks_list)={type(face_landmarks_list)}')

    # 검출된 얼굴의 수
    print(f'2) len(face_landmarks_list)=number of faces detected={len(face_landmarks_list)}')

    # 검출된  첫 번째 얼굴의 자료형 = <class 'dict'>
    print(f'3) type(face_landmarks_list[0])={type(face_landmarks_list[0])}')

    # 검출된 첫 번째 얼굴의 key를 list로 변환하고 이를 출력해 본다. 9 혹은 3개의 랜드마크가 리스트로 출력된다.
    person0_key_list = list(face_landmarks_list[0].keys())
    print(f'4) face_landmarks_list[0].keys()={person0_key_list}')
    key_list9 = ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip']
    key_list3 = ['nose_tip', 'left_eye', 'right_eye']

    # 검출된 첫 번째 얼굴에서 'left_eye'에 해당하는 좌표를 출력한다. => (x, y)로 이루어진 좌표점의 리스트이다.
    print(f"5) face_landmarks_list[0]['left_eye']={face_landmarks_list[0]['left_eye']}")

print("\n9 detected landmark details:")
print_landmark_info(face_landmarks_list_large)



# Draw all detected landmarks:
dots = 0
for face_landmarks in face_landmarks_list_large:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_large, p, 3, (0, 255, 0), -1)
            dots += 1
print(f'\n1.1 num of face_landmarks.keys()={len(face_landmarks.keys())}, Num of dots in list68={dots}')
# 1.1 num of face_landmarks.keys()=9, Num of dots in list68=72

print(1.2, face_landmarks_list_large)
# 1.2 [{'chin': [(113, 251), (111, 283), (115, 315), (122, 346), (136, 376), (154, 402), (177, 425), (203, 442), (231, 447), (260, 442), (285, 426), (306, 403), (323, 377), (334, 347), (340, 315), (343, 282), (343, 251)], 
# 'left_eyebrow': [(123, 223), (140, 211), (163, 208), (185, 211), (206, 220)],
# 'right_eyebrow': [(240, 221), (263, 212), (288, 209), (312, 211), (332, 223)],
# 'nose_bridge': [(225, 249), (225, 272), (225, 295), (226, 319)],
# 'nose_tip': [(201, 337), (213, 340), (226, 343), (239, 339), (252, 336)],
# 'left_eye': [(144, 248), (158, 239), (175, 240), (188, 254), (173, 255), (156, 254)],
# 'right_eye': [(262, 254), (276, 240), (293, 239), (308, 248), (295, 254), (278, 255)],
# 'top_lip': [(185, 377), (200, 370), (216, 364), (226, 367), (238, 364), (255, 370), (274, 377), (267, 378), (238, 378), (227, 380), (215, 379), (192, 378)],
# 'bottom_lip': [(274, 377), (257, 391), (240, 399), (228, 400), (215, 398), (200, 391), (185, 377), (192, 378), (215, 381), (227, 382), (239, 380), (267, 378)]}]



# ---------------------------------------------------------------------------------------------
# 단계 2: small model로 검출한다.
# 얼굴의  주요 랜드마크 3개를 점으로 마킹한다.
# large: 9개의 key(landmark)가 검출. small: 3개(양눈, 코)의 key가 검출
# ---------------------------------------------------------------------------------------------


# Detect 5 landmarks:
face_landmarks_list_small = face_recognition.face_landmarks(rgb, None, "small")
print("\n3 detected landmark details:")
print_landmark_info(face_landmarks_list_small)

# Draw all detected landmarks:
dots = 0        # number of dots
for face_landmarks in face_landmarks_list_small:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_small, p, 3, (0, 255, 0), -1)
            dots += 1
print(f'\n2.1 num of face_landmarks.keys()={len(face_landmarks.keys())}, Num of dots in list5={dots}')
# 2.1 num of face_landmarks.keys()=3, Num of dots in list5=5

# Print detected landmarks:
print(2.2, face_landmarks_list_small)
# 2.2 [{'nose_tip': [(227, 343)],
# 'left_eye': [(145, 248), (191, 253)],
# 'right_eye': [(307, 248), (262, 252)]}]


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Lab. 1: Facial landmarks detection using face_recognition", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(image_large, "9 facial landmarks", 1)
show_img_with_matplotlib(image_small, "3 facial landmarks", 2)



# ======================================================================================================================
# Lab 2: 눈을 검출하고 지정하는 랜드마크를 선으로 표기한다.
# 미션 - chin을 지정하는 랜드마크를 선으로 표기하시오.
# ======================================================================================================================
plt.figure(num='Lab. 2')
plt.suptitle("Lab. 2: Select landmark", fontsize=14, fontweight='bold')
num_faces = len(face_landmarks_list_large)     # 검출된 얼굴의 수

"""
# Print detected landmarks: 자료형과 그 내부 값을 출력해 본다.
print(f'1) type(face_landmarks_list_large)={type(face_landmarks_list_large)}, number of faces detected={num_faces}')
print(f'2) type(face_landmarks_list_large[0])={type(face_landmarks_list_large[0])}')
print(f'3) face_landmarks_list_large[0].keys()={face_landmarks_list_large[0].keys()}')
print(f"4) face_landmarks_list_large[0]['left_eye']={face_landmarks_list_large[0]['left_eye']}")
print(f"5) type(face_landmarks_list_large[0]['left_eye'])={type(face_landmarks_list_large[0]['left_eye'])}")
# key list=dict_keys(['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 'left_eye', 'right_eye', 'top_lip', 'bottom_lip'])
"""

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
color = (255, 0, 0)
thk = 2         # thickness
for i in range(num_faces):

    # 1) 왼쪽 눈은 중심을 구해 cyan 색 원을 그려본다.
    left_eye = face_landmarks_list_large[i]['left_eye']
    #print(type(left_eye), left_eye)     # list, [(55, 73), (57, 73), (58, 74), (60, 75), (58, 75), (56, 74)]
    pt_l = np.array(left_eye)
    #print(type(pt_l), pt_l)
    avg_l = np.mean(pt_l, axis=0)       # 각 좌표 축, (x, y)에 따라 평균을 구한다.
    print(type(avg_l), avg_l.shape, avg_l)
    pt_l  = (int(avg_l[0]), int(avg_l[1]))  # 정수로 변환 후 튜플 좌표로 변환한다.
    print(pt_l)
    cv2.circle(gray3ch, pt_l, 12, color=(0, 255, 255), thickness=-1)
    #cv2.drawContours(gray3ch, [pt_l], -1, color, thk)           # drawContours()이나 polylines() 함수나 결과는 같음.

    #cv2.drawContours(gray3ch, [pt_r], -1, color, thk)
    #gray3ch = cv2.polylines(gray3ch, [pt_l], isClosed=True, color=color, thickness=thk)

    # 2) 오른 쪽 눈은 contour을 그리듯이 polyline()으로 그려본다.
    right_eye = face_landmarks_list_large[i]['right_eye']
    pt_r = np.array(right_eye)
    avg_r = np.mean(pt_r)
    gray3ch = cv2.polylines(gray3ch, [pt_r], isClosed=True, color=color, thickness=thk)

plt.imshow(gray3ch)
plt.title('Lab. 2: Detected Face(s) with eyes marked')
plt.axis('off')
plt.show()
#cv2.imshow('Detected Facese with eyes markded', gray3ch); cv2.waitKey()
exit(0)


lm = face_landmarks_list_large[0]
print(f'key list={lm.keys()}')
pt_left_eye = lm['left_eye']
print(pt_left_eye)
exit(0)
