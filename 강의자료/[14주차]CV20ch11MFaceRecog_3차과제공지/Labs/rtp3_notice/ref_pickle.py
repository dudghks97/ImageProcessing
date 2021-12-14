"""
1. 개요
    pickle 모듈은 객체를 저장하고 다시 읽어들이는데 매우 편기한 기능을 제공한다.
    본 예제에서는 영상 데이터와 기타 정보들을 파일로 저장하고 다시 복원하는 실험을 수행한다.

2. 기능
    pickle 모듈을 이용해 직렬화, 역직렬화 작업을 수행한다.
        직렬화 (Serialization) : 객체를 파일로 내보냄
        역직렬화 (Deserialization) : 파일로 내보낸 객체를 다시 읽어들임

3. 참고
    2장 03 Non-image data persistence using numpy.py와 관계 있음
    본 예제는 이를 보봔하여 위 예제 보다 더 포괄적인 처리 기법을 제시한다.

"""


import numpy as np
import cv2 as cv
import pickle

# ========================================================================================================
# 단계 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ========================================================================================================
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = 'd:/Work/StudyImages/Images/'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
Name = 'colorbar_chart.jpg'
Name = 'lenna.bmp'
Name = 'monarch.bmp'
FullName = Path + Name

# ========================================================================================================
# n단계 1 : 영상 파일을 열어 화면에 출력하기
# ========================================================================================================
# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
image = cv.imread(FullName, cv.IMREAD_UNCHANGED)      # IMREAD_COLOR라면 생략가능
print('image.shape=', image.shape)
assert image is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
cv.imshow('image from ' + FullName, image)


# ========================================================================================================
# 단계 2 : 객체를 파일에 저장하기
# ========================================================================================================
with open("tmp_object.bin", "wb") as file:
    pickle.dump(image, file)

# ========================================================================================================
# 단계 3 : 객체를 파일에서 불러오기
# ========================================================================================================
with open("tmp_object.bin", "rb") as file:
    img2 = pickle.load(file)

# ========================================================================================================
# 단계 4 : 객체 내용 확인하기
# ========================================================================================================
cv.imshow('image from pickle file', img2)
print('img2.shape=', img2.shape)
cv.waitKey()
cv.destroyAllWindows()

# ========================================================================================================
# 단계 5 : 영상을 포함해 여러 개의 객체를 저장하고 로드하기
# ========================================================================================================

# 1) 처리를 단순화할 함수들의 정의
def saveObjects(file_name, data):       # data를 지정한 이름의 파일 file_name에 저장하기
    with open(file_name, "wb") as file:
        pickle.dump(data, file)

def loadObjects(file_name):             # 지정한 파일, file_name에서 읽은 객체를 반환하기
    with open(file_name, "rb") as file:
        return(pickle.load(file))

# 2) 저장할 객체들... 리스트 형으로 묶었다.
data = [image, image.shape, image.dtype, 'This is a string']
# data = [img, descriptor, label]

# 3) 객체들을 저장하고 다시 로드해 본다.
saveObjects('tmp.bin', data)
image1, shape1, dtype1, str1 = loadObjects('tmp.bin')

# 4) 결과를 확인한다.
print('shape1=', shape1)
print('dtype1=', dtype1)
print('str1=', str1)
cv.imshow('image from a objects file', image1)
cv.waitKey()
cv.destroyAllWindows()
