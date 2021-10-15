"""
1. 프로그램의 기능 - 영상에 대한 시그모이드 변환을 시행한다.
    1) 지정된 시그모이드 파라미터에 대한 LUT를 제작한다.
    2) 위 LUT를 활용하여 지정된 시그모이드 커브로 밝기를 변환한다.

2. 프로그램의 목표
    1) 시그모이드 변환의 특성(장점과 한계)을 이해한다.
    2) 시그모이드 함수의 3개 파라미터(m, w, E)의 제어 특성을 이해한다.

3. 프로그램에서 점검 포인트
    1) 시그모이드 함수의 용도를 이해한다.

4. 미션
    위 주어진 기능 제시에 따라 트랙바로 3개((m, w, E)) 파라미터를 제어하여
    입력 영상에 대해 시그모이드 변환을 행하는 프로그램을 설계하시오.

5. 질문
    시그모이드 변환의 장점과 한계에 대하여 기술하시오.

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'D:/work/StudyImages/images/'
#Name = 'bone(m).bmp'; # 시그모이드 부적절
#Name = 'Fig0303(a)(breast).tif'
#Name = 'Fig0306(a)(bone-scan-GE).tif'
#Name = 'CarHeadLightGlare.jpg'
#Name = 'over_exposure.jpg'
#Name = 'bukak.jpg'
#Name = 'bk_gate.jpg'
#Name = 'bk_lobby.jpg'   # 시그모이드 부적절
#Name = 'tiffany.bmp'; mid = 0.7; slope = 8
#Name = 'tank.bmp'; mid = 0.7; slope = 8
#Name = 'lenna.tif'; mid = 0.5; slope = 8
#Name = 'unnamed.jpg'
#Name = 'monarch.png'; mid = 0.5; slope = 8
Name = 'foggy.jpg'
Name = 'ele_book.jpg'
#Name = 'tiffany.bmp'

FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
# =============================================================================================


# =============================================================================================
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾼다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.
b, g, r = cv.split(img); img = cv.merge([r,g,b]);
#imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.

# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();

# 감마 변환을 칼라 영상의 각 채널에 공통적으로 적용해도 된다. 모노 변환루틴 주석처리함.
#img=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#plt.subplot(132); plt.imshow(img, cmap ='gray')
#plt.title('Gray Image'); plt.xticks([]), plt.yticks([])

#=============================================================================================

def open_figure(win_name='', fig_size=(12, 8), sub_title='', font_size=10):
    fig = plt.figure(num=win_name, figsize=fig_size)
    if sub_title == '':
        sub_title = win_name
    plt.suptitle(sub_title, fontsize=font_size, fontweight='bold')
    #fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
    #fig.patch.set_alpha(0.3)

    # 원한다면 창을 최대화하여 그림을 출력한다.
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    return fig



import sys
eps = sys.float_info.epsilon
print('epsilon=', eps)

# m = mean value. 0~1. normally 0.5.
# w = weight.  0(no strength)~1(max strength)
# E = slope. weak ~ strong.

def sigmoid_table(m=0.5, w=0.5, E=8):
    #print(m, w, E)
    r = np.arange(0, 256) / 255.0
    s = (w / (1 + (m/(r+eps))**E)) + (1-w) * r
    return (255*r).astype(np.uint8), (255*s).astype(np.uint8)


#아래 내용을 시그모이드 함수에 맞추어 직접 수정하시오.........
#또한 트랙바를 설치하고 제어하는 루틴을 추가하시오......

# 실습 1 ======================================================================================
# 변환 함수를 만들어 영상에 대해 감마 변환을 행한다.


"""
# 원본 영상 출력
plt.figure(num='Sigmoid Transform')
#plt.subplot(221); plt.imshow(imgC)      # 입력받은 대로 출력
plt.subplot(331); plt.imshow(imgC)      # 입력받은 대로 출력
plt.title('Original(color)')
plt.axis('off')     # 테두리 없음. 유사한  동작 => plt.xticks([]), plt.yticks([])
"""

open_figure(win_name='Transformed Images using various sigmoid weights', fig_size=(12, 7), font_size=18)


# 방법 a: 테이블을 바탕으로 LUT 참조로 변환하던가(방법 a), OpenCV에서 제공하는 LUT 함수를 이용하여 변환(방법 b)한다.
# dst=cv.LUT(src, lut)
# Performs a look-up table transform of an array.
# The function LUT fills the output array with values from the look-up table.
# Indices of the entries are taken from the input array.
# That is, the function processes each element of src as follows:
# dst(I)←lut(src(I) + d)
#   d=0, if src has depth CV_8U
#   d=128, if src has depth CV_8S

mids = [0.5, 0.3, 0.5, 0.5, 0.6, 0.6]
#mids = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
weights = [0, 0.5, 0.4, 0.8, 0.4, 0.8]
slopes = [0, 8, 8, 8, 8, 8]
#slopes = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
for i, (mid, weight, slope) in enumerate(zip(mids, weights, slopes)):
    _, table = sigmoid_table(m=mid, w=weight, E=slope)
    plt.subplot(231+i)
    plt.imshow(table[img])                  # 방법 a) 픽셀 값이 인덱스로 사용되는 LUT 이용 변환
    #plt.imshow(cv.LUT(img, table ))        # 방법 b) OpenCV 함수, LUT 이용 변환
    plt.title(f'm={mid}, w={weight}, E={slope}', fontsize=15)
    plt.axis('off')  # 같은 동작 => plt.xticks([]), plt.yticks([])

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.show()
#plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
exit(0)
#=============================================================================================
