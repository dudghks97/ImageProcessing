"""
아래 예제를 기반으로 수정함.
    1장- 06 Drawing 2d primitives_ markers, lines, ellipses, rectangles and text.py

개요
    p, l, r, e, t 문자를 입력하면 활성화된 영상 화면에 해당 그리기 혹은 문자 출력을 수행한다.
    이때 사용되는 파라미터는 랜덤값에 의존하기 때문에 같은 동작이라도 반복하면 가능하다.
    이때 c는 화면을 지우는 용도로 사용하고 종료 할 때는 esc 키를 사용한다.

점검 포인트
    OpenCV 창에서 키보드 입력 받은 방법을 익힐 수 있다.
        waitKey()로 받은 값을 ord() 함수를 취한 ASCII 문자와 비교한다.

"""

import cv2
import cv2 as cv
import numpy as np, random

#===============================================================================
# 단계 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
#===============================================================================
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = 'd:/Work/StudyImages/Images/'       # 이런 방식으로 사용 가능
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
Name2= 'colorbar_chart.jpg'
Name = 'lenna.bmp'
Name = 'monarch.bmp'
FullName = Path + Name
FullName2 = Path + Name2

"""
# 파이썬 연습 ----------
print("Python Exercise...")
for _ in range(10):             # 구태여 range()의 값이 필요가 없고, 단지 loop가 필요할 때
    print('A', end=" ")

print("\nLoop1")
[print('A', end=" ") for _ in range(10)]

print("\nLoop2")
for k in ['A' for _ in range(10)]:
    print(k, end=" ")

print("\nLoop3")
[print(i, end=" ") for i in range(10)]

print("\nLoop 4")
a = [i for i in range(10)]
print(type(a))
print(a)

exit()
"""


# ========================================================================================================
# 단계 1 : 영상 파일을 열기
# ========================================================================================================
# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
image = cv.imread(FullName, 1)
assert image is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

image_to_show = image.copy()
#image_to_show = np.copy(image)      # 이것도 사용 가능

w, h = image.shape[1], image.shape[0]

# ========================================================================================================
# 함수 정의: 전역변수 w, h의 범위에 있는 2개의 정수 값을 반환하는 함수.
# 영상(x, h)의 가로 x 세로 영역 안에 있는 랜덤 (x,y) 좌표로 활용예정
# random.randrange(5)       # 0~4까지의 임의의 정수를 반환.
# random.randrange(1, 7)    # 1~6까지의 임의의 정수를 반환.
# random()                  # 0부터 1 사이의 임의의 float
# ========================================================================================================
def rand_pt(mult=1.0):
    return (random.randrange(int(w*mult)),
            random.randrange(int(h*mult)))

# ========================================================================================================
# 메인루틴: 키보드 문자를 입력하여 해당 동작 수행하기
# p : circle 함수를 이용하여 10개의 파란 색 점(반지름 3)을 그린다.
# l : line 함수를 이용하여 초록 색 선을 1개 그린다.
# r : rectangle() 함수를 이용하여 붉은 색 사각형을 그린다.
# e : ellipse() 함수를 이용하여 남색 타원형을 그린다.
# esc: 프로그램 종료
# ========================================================================================================

msg = """
\n다음 글자 선택에 따라 화면의 임의 위치에 도형을 그리거나 문자를 쓴다. 
# p : circle 함수를 이용하여 10개의 파란 색 점(반지름 3)을 그린다.
# l : line 함수를 이용하여 초록 색 선을 1개 그린다.
# r : rectangle() 함수를 이용하여 붉은 색 사각형을 그린다.
# e : ellipse() 함수를 이용하여 남색 타원형을 그린다.
# t : text 'OpenCV'를 임의 색상으로 화면에 쓴다.
# c : 화면에서 그린 도형, 문자를 모두 지운다. 이후 새로 그릴 수 있다. 
# esc: 프로그램 종료
"""
print(msg)
finish = False
while not finish:
    cv2.imshow("result", image_to_show)
    key = cv2.waitKey(0)
    if key == ord('p'):
        for pt in [rand_pt() for _ in range(10)]:
            cv2.circle(image_to_show, pt, 15, (255, 0, 0), -1)
    elif key == ord('l'):
        cv2.line(image_to_show, rand_pt(), rand_pt(), (0, 255, 0), 3)
    elif key == ord('r'):
        cv2.rectangle(image_to_show, rand_pt(), rand_pt(), (0, 0, 255), 3)
    elif key == ord('e'):
        #cv2.ellipse(image_to_show, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 0), 3)
        #cv2.ellipse(image_to_show, rand_pt(), (100, 50), random.randrange(360), 0, 360, (255, 255, 0), 3)
        cv2.ellipse(image_to_show, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 0), 3)
    elif key == ord('t'):
        cv2.putText(image_to_show, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (random.randrange(255), random.randrange(255), random.randrange(255)), 3)
    elif key == ord('c'):
        image_to_show = np.copy(image)
    elif key == 27:     # check if escape input
        finish = True
