"""
1. 프로그램의 기능
    * Eg2_BinaryEdge.py와 같은 동작을 수행한다.
    * 여기서는 트랙바 콜백 함수는 하는 일이 없다.
    * 여기서는 cv.getTrackbarPos()함수를 이용하여 메인 루틴에서 트랙바의 상태를 읽어들어 해당 처리를 수행하였다.

    1차 미분 필터의 공간 필터링 결과를 다음과 같이 이진화하여 표현한다.
    에지 영상은 검은 색으로 배경은 흰색으로 처리한다.
    1) 트랙바를 이용하여 임계 값을 결정한다.
    2) 트랙바를 이용하여 양수 에지를 표현할 것인지 결정한다.
    3) 트랙바를 이용하여 음수 에지를 표현할 것인지 결정한다.
    4) 트랙바를 이용하여 x축 에지만 출력(1), y축 에지만 출력(2), 혹은 모두 출력(3)하게 할 수 있다.

2. 프로그램의 목표
    1) + 성분의 에제 혹은 - 성분의 에지가 의미하는 바가 무엇인가? -> patterns.jpg 영상에서 명확히 드러남.
    2) 1차 미분의 임계작업으로 이진 에지 영상을 얻을 때의 한계점은 무엇인가?

3. 프로그램에서 점검 포인트
    1) >, < 연산으로 영상 데이터를 임계화 작업으로 이진 결과를 얻을 수 있다.
    2) 그러나 그 결과를 cv.imshow()로 출력할 수는 없다. -> 방법은 무엇인가?
    3) 임의의 영상, img와 같은 크기의 부울(bool) 영상 타입의 데이터를 선언할 수 있는가?

4. 미션
    1) 0~1로 정규화된 칼라 영상에 대해 같은 기능을 구현할 수 있도록 프로그램을 수정하시오.
    2) 수평 에지, 수직 에지를 선택적으로 표시할 수 있는 트랙바 2 개를 추가하시오.

"""


import numpy as np
import cv2 as cv
# 만약 matplotlib 모듈이 없다면 윈도 컴맨드 창에서 "pip install matplotlib" 명령을 수행한다. 
from matplotlib import pyplot as plt



# -------------------------------------------------------------------------------------------
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'graphicShapes.png'
#Name = 'circles_rects.png'
#Name = 'circles_rects_2.png'
#Name = 'gray2.jpg';         # Threshold 345로 관찰 요망.
Name = 'lenna.tif'
#Name = 'GraphicShapes.png';     # 고품질 그래픽스 영상
#Name = 'Mondrian.jpg'

FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

cv.imshow('Original', img)
# -------------------------------------------------------------------------------------------

# 단계 1 ------------------------------------------------------------------------------
# 에지 영상은 흑백 영상으로 관찰하는 것이 바람직하다.
# 입력 영상을 1채널의 흑백 gray 영상으로 변환한다.
# dst=cv.cvtColor(src, code[, dst[, dstCn]]	)
#       여기서, code= cv.COLOR_RGB2GRAY, cv.COLOR_BGR2GRAY. 주의!  적정한 것을 선택.
img = cv.cvtColor(src=img, code=cv.COLOR_BGR2GRAY)
print("cv2: type(img)=", type(img), " img.dtype=", img.dtype, "img.shape=", img.shape)
cv.imshow('Original in gray', img)
#cv.waitKey()


# 단계 2 ------------------------------------------------------------------------------
# 커널의 종류와 그 내부 계수 값을 정의한다.

# 로베르츠 에지 연산자
RobertsSlash = [[0, 0, 0], [0, -1, 0], [0, 0, 1]]       # / 방향의 에지를 찾아내는 로베르츠 연산자.
RobertsBackSlash = [[0, 0, 0], [0, 0, -1], [0, 1, 0]]   # \ 방향의 에지를 찾아내는 로베르츠 연산자.

# 프리위트 에지 연산자
#PrewittHorizontal = [1 0 -1; 1 0 -1; 1 0 -1] ;	# 수평방향의 1차 미분
#PrewittVertical = [1 1 1; 0 0 0; -1 -1 -1];	# 수직방향의 1차 미분

# 소벨 에지 연산자(밝은 곳에서 어두운 곳으로 올때 + 에지 발생)
#SobelHorizontal = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]	# 수평방향의 1차 미분
#SobelVertical = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]] 	    # 수직방향의 1차 미분


# 소벨 에지 연산자(어두운 곳에서 밝은 곳으로 올때 + 에지 발생, OpenCV Sobel() 함수가 사용)
SobelHorizontal = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]	    # 수평방향의 1차 미분
SobelVertical = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] 	    # 수직방향의 1차 미분


# 프라이첸 에지 연산자
#FriHorizontal = [1 0 -1; sqrt(2) 0 -sqrt(2); 1 0 -1] ;   # 수평방향의 1차 미분
#FriVertical = [1 sqrt(2) 1; 0 0 0; -1 -sqrt(2) -1] ;      # 수직방향의 1차 미분


kerH = np.array(SobelHorizontal, dtype=float )     # print(type(kerH))
kerV = np.array(SobelVertical, dtype=float )       # print(type(kerV))

#커널을 출력하되 원하는 정밀도로 출력하게 한다. >>> help(np.set_printoptions)
np.set_printoptions(precision=1, floatmode='fixed'); print(kerH, '\n', kerV)  # 소수점 이하 1째 자리까지 출력.


# 단계 3 ------------------------------------------------------------------------------
# 트랙바 콜백 함수를 정의하고 트랙바 초기화 함수를 호출한다.

def nothing(x):
    pass

cv.imshow('image',img)
cv.createTrackbar('Threshold', 'image', 30, 500, nothing)   # Threshold 초기 값= 30. 최대 500
cv.createTrackbar('+ edge', 'image', 1, 1, nothing)  # + 에지 초기 선택값 =1, 0=출력 안함. 1=+에지만 출력
cv.createTrackbar('- edge', 'image', 0, 1, nothing)  # - 에지 초기 선택값 =0, 0=출력 안함. 1=-에지만 출력한다.
# x축 에지, y축 에지를 선택한다. 초기 선택값 =1. 0=선택안함. 1=x만 출력. 2=y만 출력, 3=xy 모두 출력
cv.createTrackbar('x=1y=2', 'image', 1, 3, nothing)


# 단계 4 ------------------------------------------------------------------------------
# 2차원 correlation 연산을 수행한다.
# dst= cv.filter2D(	src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]	)
edgeH = cv.filter2D(src=img, ddepth=cv.CV_32F, kernel=kerH); print('edgeH.dtype=',edgeH.dtype)
edgeV = cv.filter2D(src=img, ddepth=cv.CV_32F, kernel=kerV); print('edgeV.dtype',edgeV.dtype)


# 단계 5 --------------------------------------------------------------------------------------------
# 트랙바의 선택에 따른 결과를 화면에 출력한다.
# 참고: 무한 루프를 돌면서 화면 갱신을 끊임없이 반복하기 때문에 CPU 자원 활용 관점에서 바람직하지 않은 설계이다.
# 미션: 마우스 동작이 일어날 때만 화면 갱신을 시도하도록 바꾸는 프로그램을 수정해 보시오..
# ----------------------------------------------------------------------------------------------------
while (1):
    imgEdge = np.zeros(img.shape, 'bool')
    k = cv.waitKey(1) # & 0xFF
    if k != -1:     # break if any key in input
        break
    Threshold = cv.getTrackbarPos('Threshold', 'image')

    edgeHplus = edgeH > Threshold       # 가로 에지 + 성분. bool image.
    edgeHminus = edgeH < -Threshold     # 가로 에지 - 성분. bool image.
    edgeVplus = edgeV > Threshold       # 세로 에지 + 성분. bool image.
    edgeVminus = edgeV < -Threshold     # 세로 에지 - 성분. bool image.

    # 주의 !!!! : OpenCC-Python에서는 bool type 데이터는 imshow()로 표시할 수 없다.
    # 추천 해결법: 아래 사례처럼 1.0을 bool 데이터에 곱해서 0~1 범위의 부동소수 데이터로 표현한다.
    # tmp = 1.0 * edgePlus; cv.imshow('edgePlus', tmp); print('tmp.dtype=',tmp.dtype )

    if cv.getTrackbarPos('x=1y=2', 'image') == 0:       # X축 에지, y축에지 모두 선택 안함.
        edgePlus = imgEdge
        edgeMinus = imgEdge
    if cv.getTrackbarPos('x=1y=2', 'image') == 1:       # X축 에지만 관찰하고 싶다
        edgePlus = edgeHplus
        edgeMinus = edgeHminus
    if cv.getTrackbarPos('x=1y=2', 'image') == 2:       # y축 에지만 관찰하고 싶다
        edgePlus = edgeVplus
        edgeMinus = edgeVminus
    if cv.getTrackbarPos('x=1y=2', 'image') == 3:       # X, y축 에지 모두 관찰하고 싶다
        edgePlus = edgeHplus + edgeVplus
        edgeMinus = edgeHminus + edgeVminus
    if cv.getTrackbarPos('+ edge', 'image') == 1:       # + 에지 성분만 관찰
        imgEdge = imgEdge | edgePlus
    if cv.getTrackbarPos('- edge', 'image') == 1:       # -에지 성분만 관찰
        imgEdge = imgEdge | edgeMinus

    # bool 영상은 출력이 안되므로 부동소수 영상으로 만들어 출력한다. 편의상 역상으로 변환하여 출력한다.
    cv.imshow('image', 1- 1.0 * imgEdge)

