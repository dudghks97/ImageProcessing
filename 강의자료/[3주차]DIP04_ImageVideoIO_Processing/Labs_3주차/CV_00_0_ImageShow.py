"""

개요
    OpencCV 함수 기반으로 영상 파일을 읽어들여 화면에 출력한다.

동작
    1. 영상 파일을 읽어,
    2. 화면에 출력 출력하고
    3. 다른 이름(형식)의 파일로 저장하기

함수
    1. 영상 파일("abc.jpg") 읽어 영상 어레이(img)에 반환하기: img=cv.imread("abc.jpg", 열기_모드)
    2. 영상 어레이(img)를 지정된 이름("win")의 타이틀에 출력하기: imshow("win", img)
    3. 키입력을 무한 혹은 500[ms]동안 기다리기: waitKey()=waitKey(0) or waitKey(500)

질문
    1. 영상 파일을 읽을 때 칼라 파일을 1채널의 모노 파일로 읽어 들이는 방법은?
    2. 1채널의 모노 영상을 3채널의 칼라 데이터로 읽어 들이는 방법은?

"""

import cv2 as cv

# 현재의 위치에서 상위 폴더의 data 폴더에는 이번 시간 다룰 샘플 영상과 비디오 동영상을 저장하였다.
a = cv.imread('../data/monarch.bmp')           # 영상 파일이 현재의 폴더에 있어야 한다.

print(f'type(a)={type(a)}, a.shape={a.shape}, a.dtype={a.dtype}')
#print('type(a)={}, a.shape={}, a.dtype={}'.format(type(a), a.shape, a.dtype))
cv.imshow('test1', a)                    # 2) 영상 보이기. test=창의 이름
cv.waitKey()    # 무한정 키를 기다리기..
cv.imwrite('tmp_monarch.png', a)        # 3) 영상 저장. jpg, png 등 형식 지정 가능

a = cv.imread('../data/smooth.jpg')
cv.imshow('test2', a)                    # 2) 영상 보이기. test=창의 이름
cv.waitKey(5000)                        # 2000[ms]=2초 기다리기