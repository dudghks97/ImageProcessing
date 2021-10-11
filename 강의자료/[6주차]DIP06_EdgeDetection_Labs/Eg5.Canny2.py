"""
1. 프로그램의 기능
    Canny() 함수를 이용하여 영상의 에지를 검출한다.
    3개의 파라미터를 Trackbar를 이용하여 조정하면서 검출 효과를 검토할 수 있게 하였다.
    트랙바 이벤트가 일어날 때만 영상을 업데이트하도록 조정하였다.
    Canny() 함수의 파라미터 [T_low, T_high] 조정에서 "T_low < T_high" 조건을 추가하였디.
        트랙바 자체는 조건에 맞지 않으면 움직이지 않는다.
        그러나 트랙바의 출력 값이 범위를 넘어서는 값이 출력되는 것은 막을 수가 없었다.
            setTrackbarPos() 함수를 위치를 정정하면 트랙바의 위치는 정정되지만, 출력 값이 조건을 벗어나는 값을 출력하는 막을 수가 없다.
    현재 설정된 임계값과 사이즈를 트랙바를 움직일 때마다 계속 print() 하도록 하였다. 예: threshold=[0, 199], Size=7

2. 프로그램의 목표
    Canny() 함수의 파라미터 [T_low, T_high]의 에지 검출에 미치는 영향을 이해한다.
    sigma의 변화가 검출되는 에지에 미치는 영향을 이해한다.

3. 프로그램에서 점검 포인트
    트랙바를 제어하는 함수를 사용하였다.
        cv.createTrackbar('트랙바_이름', '창_이름', 슬라이더_초기_위치, 슬라이더_최댓값, 콜백함수)
    트랙바의 현재 위치를 알아내는 함수를 사용하였다.
        위치_값 = cv.getTrackbarPos('트랙바_이름', '창_이름')
4. 미션
   L2gradient를 False, True 중 바꾸어서 설정할 때의 차이점에 대하여 의견을 제시하여 보자.

참고: https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html

5. Canny() 함수의 용법
    https://docs.opencv.org/4.5.2/dd/d1a/group__imgproc__feature.html#ga04723e007ed888ddf11d9ba04e2232de
    edges =	cv.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])
    반환값 edges는 uint8 type.

"""

# ----------------------------------------------------------------------------------------------------------------------
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'graphicShapes.png'
#Name = 'circles_rects.png'
#Name = 'circles_rects_2.png'
Name = 'lenna.tif'
#Name = 'GraphicShapes.png'     # 고품질 그래픽스 영상
#Name = 'Mondrian.jpg'
#Name = 'monarch.jpg'

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


FullName = Path + Name
img = cv.imread(FullName, 0)     # 주의 !! 모노로 읽어냄. Canny() 함수는 1채널만 처리함.

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

# 전역 변수 설정
T_low = 100        # T_low
T_high = 200       # T_high
Size = 1           # Canny() apertureSize. 내부 연산에 활용되는 소벨 연산자의 크기는 2*Size+1로 산정된다.
Touched = False     # 트랙바를 건드렸다면 True. 이 때만 연산을 다시 한다.

def callback_th_min(x):       # thresold를 조절한다. T_low 설정값이 이곳으로 전달된다.
    global T_low, Touched       # 이 함수 안에서 변경될 변수만 전역변수로 선언한다.
    # "T_low < T_high" 조건을 만족하지 않으면 조절하지 않는다.
    if x < T_high:          # "T_low < T_high" 조건이 만족하는가?
        T_low = x           # 정상적인 값이므로 허용.
        Touched = True
    else:
        cv.setTrackbarPos('T_low', 'Canny', T_low)      # 허용되지 않으므로 원상 복귀
    print(f"\rthreshold=[{T_low}, {T_high}], Size={Size}", end='')

def callback_th_max(x):       # thresold를 조절한다. T_high 설정 값이 이곳으로 전달된다.
    global T_high, Touched      # 이 함수 안에서 변경될 변수만 전역변수로 선언한다.
    # "T_low < T_high" 조건을 만족하지 않으면 조절하지 않는다.
    if x > T_low:          # "T_low < T_high" 조건이 만족하는가?
        T_high = x       # 정상적인 값이므로 허용.
        Touched = True
    else:
        cv.setTrackbarPos('T_high', 'Canny', T_high)      # 허용되지 않으므로 원상 복귀
    print(f"\rthreshold=[{T_low}, {T_high}], Size={Size}", end='')

def callback_size(x):       # x에는 apertureSize를 결정하기 위한 정보가 유입된다.
    global Size, Touched
    Size = x * 2 + 1
    #if Size == 1:
    #    Size = 3
    #cv.setTrackbarPos('Size', 'Canny', Size)  # 창이름 Canny의 Trackbar size의 현재의 슬라이더 위치=3으로 이동
    print(f"\rthreshold=[{T_low}, {T_high}], Size={Size}", end='')
    Touched = True


cv.imshow('Original', img)
cv.namedWindow('Canny')

# 1) trackbar install for threshold1:	first threshold for the hysteresis procedure.
cv.createTrackbar ('T_low', 	    # 트랙바 앞에 표시될 트랙바의 이름
'Canny',	        # 트랙바가 나타날 창의 이름
T_low,			    # 시작 당시의 슬라이더의 초기 위치
512,		        # 슬라이더의 최댓값. 최솟값은 0으로 고정.
callback_th_min)	        # 슬라이더가 움직일 때 호출될 콜백 함수의 이름

# 2) trackbar install for threshold2:	second threshold for the hysteresis procedure.
cv.createTrackbar('T_high', 'Canny', T_high, 512, callback_th_max)     # 초기값 th_max

# 3) trackbar install for apertureSize:	aperture size for the Sobel operator.
cv.createTrackbar('Size', 'Canny', Size, 3, callback_size)       # 초기값 Size=3, 최대 크기 3
edges = cv.Canny(img, T_low, T_high, Size)

print("")
while(True):
    cv.imshow('Canny', edges)
    k = cv.waitKey(1) # & 0xFF
    if k != -1:     # break if any key in input
        break
    if Touched == True:         # 트랙바를 건드렸다면 에지를 다시 검출한다.
        edges = cv.Canny(img, T_low, T_high, Size)      # 새로 설정된 값으로 에지를 다시 검출한다.
        Touched = False


