"""
1. 개요: ROI: Region Of Interest 설정
    OpenCV는 영상에서 관임있는 특정 영역을 마우스 드래그로 선택하는 함수(selectROI)를 제공한다.
    * 주의 사항
        원래 ROI란 영역은 영상의 특정 영역을 변수로 지정해서 이 변수를 통해 영상전체 영역 중의 일부 영역을 액세스하기 위해 만들어졌다.
        파이썬은 일정 영역을 슬라이싱을 통해 어떤 변수를 선언하면, 그 영역이 복사되어 새로운 작은 어레이를 만들어 낸다.
        따라서, C와는 달리 이 변수의 포인터를 이용해서 특정 영역에 쓰기 동작을 할 수 없다.
        극, 파이썬에서는 "ROI = 값" 스타일의 코딩이 불가하다.
        단지. 어레이 슬라이싱을 이용해서 쓰기를 할 수 있을 뿐이다.
        -- 코딩 관점에서 ROI 지정은 읽기 동작에서만 의미가 있고, ROI에 쓰기 동작은 불편한 슬리이싱 방식으로 구형해야 한다.
        이는 당초 ROI의 취지에 어긋난 것으로 이해된다.

2. 동작 개요
    영상의 특정 영역(ROI, Region Of Interest)을 변수로 설정하여 특정 영역 데이터만 편리하게 액세스한다. -- 읽기 동작만 편리.
    OpenCV에서는 ROI를 mouse drag 동작으로 화면상에 쉽게 선택할 수 있는 selectROI() 함수를 제공한다.
        retval	=	cv.selectROI(	windowName, img[, showCrosshair[, fromCenter]]	)
        retval	=	cv.selectROI(	img[, showCrosshair[, fromCenter]]	)

3. 프로그램의 흐름
    1) selectROI() 함수를 이용해 원본 영상을 출력하고, 마우스 드랙으로 ROI 영역을 선택받는다.
        드래그 과정에서는 파란 색 4각형으로 선택하고자 하는 ROI가 표현된다.
        4각형을 잘못 그렸으면 다시 드래그를 동작을 반복하거나 'c'를 입력하여 다시 그릴 수 있다.
    2) 4각형이 다 그려졌으면, space or enter 버튼(혹은 esc 키)을 입력하면
        selectROI() 함수는 선택된 ROI의 4각형 정보를 튜플로 반환한다.
        tuple 자료(4개 원소): (좌측 상단 점의 열의 좌표, 좌측 상단의 점의 행의 좌표, 4각형 넓이, 4각형 높이)
    3) 선택된 ROI와 그 역상을 화면에 출력한다.
    4) 원본 영상의 ROI 부분을 밝게 바꾸어 화면에 출력한다.
Rect 자료 - 4각형 자료 표현

# 참고: C++에서는 ROI가 메모리의 특정 영역을 지칭(읽거나, 쓰거나)할 때 사용할 수 있다.
#       파이썬에서는 읽을 때는 ROI가 C++처럼 쓰이지만, 그 영역에 데이터를 dump할 때는 slicing operation으로 바꾸어야 하는 불편함이 있다.
#       ROI의 취지에 맞지 않는 듯...
#       C++의 사례:
#       // define image ROI at image bottom-right
#           cv::Mat imageROI(image, 
#                       cv::Rect(image.cols-logo.cols, // ROI x 좌표
#                        image.rows-logo.rows,         // ROI y 좌표
#                        logo.cols,logo.rows));        // ROI 폭, 넓이
# 
#       // insert logo. ** ROI 변수를 사용하여 쓰기가 된다.
#           logo.copyTo(imageROI);      # logo 영상을 image 영상의 일부분, roi인 imageROI에 copy한다.
#                                       # 즉, ROI 변수를 지정하여 쓰기 동작이 가능하다.
#                                       # 어떤 영역(즉, ROI)를 변수로 할당하고 그 곳에 write 할수 있다는 것이 장점이다.
#                                       # 파이썬에는 이 기능이 없는 듯 하다. 아직 못 찾았음.
#

"""

# ========================================================================================================
# 섹션 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ========================================================================================================
# Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
# Path = 'd:/StudyImages/Images/'        # 사용가능
Path = 'd:/Work/StudyImages/Images/'
Path = '../data/'

Name = 'lenna_mono.jpg'  # mono gray 영상. 1채널 영상
Name = 'RGBColors.JPG'
Name = 'colorbar_chart.jpg'
Name = 'lenna.tiff'
Name = 'monarch.bmp'

FullName = Path + Name

import cv2 as cv
import numpy as np

# 영상 파일을 읽어 들이기
img = cv.imread(FullName)
assert img is not None, f'No image file....[{Name}]!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

# ========================================================================================================

# 윈도 타이틀에 출력할 가이드
msg_str = "Drag mouse and press the space bar to select ROI.."
print(f'\n{msg_str}\n')

# ROI selector 창을 생성하고 마우스 드래그를 점검한다.
# 스페이스 혹은 엔터를 입력하면 선택한 rect 정보를 반환한다.
# 해당 창은 그대로 남아 있는다.

# 1) ROI로 선택한 영역 정보 반환받는 방법 1: 4각형으로 받기
# r = cv.selectROI(img)     # default windowName = "ROI selector", 기본적으로 십자선이 있음
# r = cv.selectROI(msg_str, img, showCrosshair=False)     # windowName = str_msg, 십자선 지우기
# Rect형 자료: (좌측 상단 점의 x 좌표, 좌측 상단 점의 y 좌표, 넓이, 높이) => x,y,w,h
# print(f'type(r)={type(r)}: {r}, type(r[0])={type(r[0])}')
# type(r)=<class 'tuple'> type(r[0])=<class 'int'>


# 2) ROI로 선택한 영역 정보 반환받는 방법 2: 개별정보 4개로 받기
selected = False
while selected == False:
    x, y, w, h = cv.selectROI(msg_str, img, showCrosshair=False)
    if x == 0 and y == 0 and w == 0 and h == 0:  # x좌표. y좌표, 넓이(가로). 높이(세로)
        print("'c' seems to be pressed...")
        continue
    else:
        print(f'ROI: x={x}, y={y}, w={w}, h={h}')
        break

# cv.destroyWindow(msg_str)     # 원하면 ROI 선택창을 닫을 수 있다.

roi = img[y:y + h, x:x + w]  # 원본 영상의 선택된 부분을 roi 변수 어레이에 복사해 넣는다.
print(f'roi.shape={roi.shape}')  # (row, clos, ch)

cv.imshow('roi', roi)
cv.waitKey(0)

my_roi = 255 - roi  # 관심 영역에만 영상처리를 행한다. 여기서는 역상으로 만드는 처리.
cv.imshow('roi inverse', my_roi)
cv.waitKey()

# 두개의 영상을 가로로 이어 붙여 본다. 세로버전: vconcat())
two = cv.hconcat([roi, my_roi])
cv.imshow('hconcat image', two)
cv.waitKey()

cv.imshow('img', img)  # img의 일부 영역을 ROI로 정하였지만, ROI를 바꾸었다고 img까지 바뀌는 것은 아니다.
cv.waitKey()


# 영상 전체 영역 중 my_roi가 속한 영역에 결과를 넣는 2가지 방법
# my_roi를 img의 일부 영역에 복사해 넣어야 바뀐다.
#img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = my_roi       # 1) 4각형 튜플을 이용한 넣는다.
img[y:y + h, x:x + w] = np.clip(roi.astype(np.uint16) + 50, 0, 255)              # 2) 4가지 정보를 변수로 지정하여 넣는다.

cv.imshow('img', img)
cv.waitKey(0)
