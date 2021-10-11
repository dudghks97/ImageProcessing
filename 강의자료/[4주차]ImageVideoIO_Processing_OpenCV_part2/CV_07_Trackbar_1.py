"""
1장-05 Working with UI elements such as scrollbars in OpenCV window.py와 관계가 있음.

1. 프로그램의 기능
    트랙바를 이용해 사각형 영상의 색상을 조절한다.
    아무런 키나 입력하면 프로그램은 종료한다.

2. 프로그램의 목표
    트랙바를 설치하고 관리하는 기법을 습득한다.

3. 프로그램에서 점검 포인트
    createTrackbar()에서 슬라이딩 동작을 수행할 때 호출될 트랙바 콜백 함수를 지정한다.
    이 프로그램에서는 실제로는 콜백함수(nothing)은 하는 역할이 없다.
    --- 메인 루프에서 각 트랙바의 위치를 읽어서 적절한 처리를 하기 때문이다.

4. 현 프로그램의 문제점과 개선의 포인트: => 아를 개선한 것이 Trackbar_2.py
    1) 현재는 수정하지 하지 않은 다른 칼라까지도 while loop 안에서 무조건 새로 update한다.
    정지하고 있는 듯이 보여도 프로그램은 끊임없이 버퍼의 내용을 화면에 출력하는 작업을 지속하고 있다.
    muti-tasking 환경에서는 프로세스를 점유해서 전체적인 시스템 성능을 떨어뜨릴 가능성이 많다.
    2) 콜백함수를 호출할 때 이미 파라미터로 마우스의 위치가 전달되었지만 본 프로그램에서는 메인 루프에서
    getTrackbarPos 함수로 다시 마우스의 위치를 불필요하게 또 읽어내는 점이 불합리하게 설계되었다.

5. 주요 함수의 사용법
    1) createTrackbar 함수의 사용법 - 트랙바를 설치하고 콜백 함수를 등록한다.
    int cv::createTrackbar (const String & 	trackbarname, 	# 트랙바 앞에 표시될 트랙바의 이름
        const String & 	winname,		# 트랙바가 나타날 창의 이름
        int * 	value,			# 시작 당시의 슬라이더의 초기 위치
        int 	count,			# 슬라이더의 최댓값. 최솟값은 0으로 고정.
        TrackbarCallback 	onChange = 0,	# 슬라이더가 움직일 때 호출될 콜백 함수의 이름
        void * 	userdata = 0) 		# 콜백 함수에게 넘겨질 사용자 데이터. 이 파라미터는 파이썬에서는 지원 안함.
        파이썬에서는 메인루틴의 변수가 함수에서 그대로 쓰여질 수 있는 사실상 전역변수화되기 때문에 필요가 없어서 때문인 듯하다.

    2) TrackbarCallback 함수에 대해..
        아래 typedef 문에 의해 함수는 2개의 파라미터가 실제로 불려질 함수에 넘어가게 된다.
        typedef void(* cv::TrackbarCallback) (int pos, void *userdata)

        Parameters
            pos: current position of the specified trackbar.
            userdata: The optional parameter.

    3) 트랙바의 위치 지정 및 위치 읽기
        cv.setTrackbarPos('트랙바의 이름', '창의 이름', 지정할 위치)
        반환받을 트랙바 위치 =cv.setTrackbarPos('트랙바의 이름', '창의 이름')

6. 참조
    https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_trackbar/py_trackbar.html

* bug report
메시지: Using 'value' pointer is unsafe and deprecated. Use NULL as value pointer. To fetch trackbar value setup callback.
진단: OpenCV 4.5.3과 python 3.7이상에서 발생하는 OpenCV의 버그
조치: 해당 CPP 파일 소스를 수정하면 됨. 혹은 다음 릴리즈에서 정리될 것을 보임.
https://stackoverflow.com/questions/68449992/using-value-pointer-is-unsafe-and-deprecated-error-in-opencv-python
위 링크를 보면 이것은 정리된 버그라고 밝혀져 있음 ==> https://github.com/opencv/opencv/issues/20408

"""



import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# ========================================================================================================
# 함수: nothing(), 슬라이드 바를 움직이면 자동으로 호출되어 수행되는 함수이다.
# 여기서는 아무 일도 하지 않는다. 대신 메인 루틴에서 마우스의 위치를 명시적으로 읽어낸다.
# 원래 이 함수에는 트랙 바에서 설정된 값(위치)와 사용자 데이터(x, *userdata)가 자동으로 입력된다.
# 그러나, 여기서는 사용자 데이터를 사용하지 않아서 정의하지 않아도 된다.
# ========================================================================================================

#def nothing(x, *userdata):    # 이것이 원래 올바른 형식. 그런데 조만간 없어질 것 같음.
#    pass

#def nothing(x):                 # 이것도 됨
#    pass

#def nothing(NULL):                     # 이것도 됨.
#    pass

def nothing():                     # 이것도 됨.
    pass

# ========================================================================================================
# 단계 1. 적당한 크기의 비어있는 4각형 영상 어레이를 정의한다.
# 트랙바의 슬라이드 선택에 의해 이 어레이에 해당 색상을 채워 넣을 예정이다.
# ========================================================================================================
img = np.zeros((300, 900, 3), np.uint8)
print(img.shape)
#cv.namedWindow('image')    # <= 이것만 쓰려면 창의 크기를 지정해 주어야 트랙바의 크기가 적게 열리는 것을 방지할 수 있다.
cv.imshow('image', img)     # 확실하게 처음부터 영상 크기로 연다.

# ========================================================================================================
# 단계 2. # create trackbars for color change
# 4 개의 트랙바를 설치한다.
# 아래의 트랙바들은 슬라이드를 움직이면 모두 콜백 함수, nothing을 호출한다.
# ========================================================================================================
cv.createTrackbar('R', 	# 트랙바 앞에 표시될 트랙바의 이름
'image',	# 트랙바가 나타날 창의 이름
5,			# 시작 당시의 슬라이더의 초기 위치
255,		# 슬라이더의 최댓값. 최솟값은 0으로 고정.
nothing	    # 슬라이더가 움직일 때 호출될 콜백 함수의 이름
)

cv.createTrackbar('G', 'image', 0, 255, nothing)
cv.createTrackbar('B', 'image', 0, 255, nothing)

# create switch for ON/OFF functionality
switch = 'ON|OFF'
cv.createTrackbar(switch, 'image', 1, 1, nothing)   # 시작점 초기위치=1, 최댓값=1

# ========================================================================================================
# 단계 3. 메인 루틴
# 각 슬라이드바의 설정 위치를 읽어내어서 이에 해당하는 색상으로 영상 어레이를 채워 넣어 화면에 출력한다.
# 코딩이 단순해서 편리하긴 하지만, 슬라이드를 움직일 때마다
# 변하지 않은 다른 슬라이드의 색상까지 같은 같은 값으로 채워 넣는 불합리한 면이 있다.
# ========================================================================================================
while(1):
    cv.imshow('image', img)
    k = cv.waitKey(1)   # & 0xFF
    if k != -1:         # break if any key in input. 무언가 키를 입력한 일이 없으면 -1을 반환한다.
        break

    # retval=cv.getTrackbarPos(	trackbarname, winname)
    # getTrackbarPos returns the current position of the specified trackbar.
        # trackbarname:	Name of the trackbar.
        # winname:	Name of the window that is the parent of the trackbar.
    r = cv.getTrackbarPos('R','image')
    g = cv.getTrackbarPos('G','image')
    b = cv.getTrackbarPos('B','image')
    s = cv.getTrackbarPos(switch,'image')

    if s == 0:
        img[:] = 0      # 사각형을 검게 칠한다.
        #img = 0  # 이것은 안됨. 스칼라변수 img를 새로 선언하는 것과 같음.
    else:
        img[:] = b, g, r
        #img[:, :, 0] = b; img[:, :, 1] = g; img[:, :, 2] = r;
        #img[..., 0] = b;
        #img[..., 1] = g;
        #img[..., 2] = r;
cv.destroyAllWindows()