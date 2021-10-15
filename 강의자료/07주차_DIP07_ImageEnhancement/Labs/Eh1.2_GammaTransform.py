"""
1. 프로그램의 기능 - 영상에 대한 감마 변환을 시행한다.
    실습 1: 지정된 감마에 대한 LUT를 설계한다.
            위 LUT를 활용하여 지정된 계조값을 기준으로
                1) LUT를 참조하는 OpenCV 함수를 이용해 명암을 변환한다.
                2) 혹은 함수를 이용하지 않고, 자체적으로 직접 LUT를 참조하여 명암을 변환한다.
            편의상 gamma > 1인 경우에 대해 행한다.
    실습 2: 지정된 감마에 대한 각 계조값에 대한 변환을 영상의 모든 영역에 대해 행한다.
            매번 감마 함수를 호출하여 연산하기 때문에 시간이 더 많이 소요될 것으로 예상된다.
            편의상 gamma < 1인 경우에 대해 행한다.

참고: 수행시간 측정 결과 사례
    execution time for Lab 1: 0.07[sec.]
    execution time for Lab 2: 4.40[sec.]

2. 프로그램의 목표
    1) 감마 변환의 영상에 미치는 특성(장점과 한계)을 이해한다.
    2) LUT의 특성(장점)과 활용방법을 이해한다.

3. 프로그램에서 점검 포인트
    1) 1차 신호 함수를 파이썬 루틴으로 정의한다.
    2) 함수를 원하는 형태의 그래프로 출력하는 method를 익힌다.
    - legend, grid, axis, xlable, ylabel

4. 미션
    다음과 같은 감마 변환의 구현 방법의 소요 시간을 측정하여 그 실험 결과를 제시하시오.
    1) 변환 테이블 없이 직접 감마 변환 함수를 연산 하여 변환할 경우
    2) 감마 변환 테이블을 만든 후 아래 각 방법으로 시행할 경우
        a) OpenCV 함수, LUT 이용 변환
        b) Python의 인덱싱 생략 방법(for all value)
        c) for loop를 이용하여 각 화소별로 명시적으로 지정
    - 10개 정도의 영상을 4개의 알고리즘이 번갈아 연산하는 작업을 10회 반복한 결과에 대해 통계 작업 적용
    - FHD 화질(1920x1080x3)의 영상에 대한 적용 결과로 환산하여 비교
    - 순수하게 변환처리한 부분에 소요되는 시간만을 비교해야 함.
    - 컴퓨팅 환경에 대해 밝혀야 함. - CPU, clock speed, main Memory, GPU 보드 활용 여부 및 사양
    - 힌트: 아래와 같이 파이썬 내장 모듈 time을 활용

    import time
    start = time.time()
    # run your code
    end = time.time()
    elapsed = end - start   # 소요된 시간: ms 단위. 10**(-3)초.

5. 질문
    감마 변환의 장점과 한계에 대하여 기술하시오.

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time


#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'D:/work/StudyImages/images/'
Name = 'lenna.tif'
#Name = 'bone(m).bmp'
#Name = 'Fig0303(a)(breast).tif'
#Name = 'Fig0306(a)(bone-scan-GE).tif'
#Name = 'CarHeadLightGlare.jpg'
#Name = 'over_exposure.jpg'          # 고조도
#Name = 'tiffany.bmp'                # 고조도
#Name = 'bukak.jpg'                 # 저조도
#Name = 'bk_gate.jpg'
Name = 'bk_lobby.jpg'
#Name = 'dark1.png'

FullName = Path + Name
img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

#=============================================================================================


#=============================================================================================
# 단계 1 - # matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾼다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.
b, g, r = cv.split(img); img = cv.merge([r, g, b]);
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다.

# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();

# 감마 변환을 칼라 영상의 각 채널에 공통적으로 적용해도 된다. 모노 변환루틴 주석처리함.
#img=cv.cvtColor(img, cv.COLOR_RGB2GRAY)
#plt.subplot(132); plt.imshow(img, cmap ='gray')
#plt.title('Gray Image'); plt.xticks([]), plt.yticks([])

# ============================================================================================


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


# 실습 1 ======================================================================================
# gamma > 1: 변환 함수를 LUT로 만들어 LUT 참조를 이용해 영상에 대해 감마 변환을 행한다.
# 방법 : 지정된 감마에 대한 LUT를 설계한다.
#        위 LUT를 활용하여 지정된 계조값을 기준으로
#          1) LUT를 참조하는 OpenCV 함수를 이용해 명암을 변환한다.
#          2) 혹은 함수를 이용하지 않고, 자체적으로 직접 LUT를 참조하여 명암을 변환한다.
#        편의상 gamma > 1인 경우에 대해 행한다.
# ============================================================================================

open_figure(win_name='Gamma Transformed Images using LUT reference: gamma > 1', fig_size=(12, 8), font_size=18)
plt.subplot(221); plt.imshow(imgC)      # 입력받은 대로 출력
plt.title('Original(color)')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])

# 감마 변환 함수 정의한다. 감마를 지정하면 변환을 위한 table을 반환한다.
# https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
def gammaTrans(gamma=1.0):
    table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return table

# 방법 1: 테이블을 바탕으로 OpenCV에서 제공하는 LUT 함수를 이용하여 변환한다.
# dst=cv.LUT(src, lut)
# Performs a look-up table transform of an array.
# The function LUT fills the output array with values from the look-up table.
# Indices of the entries are taken from the input array.
# That is, the function processes each element of src as follows:
# dst(I)←lut(src(I) + d)
#   d=0, if src has depth CV_8U
#   d=128, if src has depth CV_8S

i = 2
start_t = time.time()               # 시간 측정을 위해 시작 시간 기록
#for gamma in [0.6, 0.4, 0.2]:          # 밝게 만들기
for gamma in [2, 3, 4]:             # 어둡게 만들기
    table = gammaTrans(gamma)
    plt.subplot(220+int(i))
    plt.imshow(cv.LUT(img, table))  # 방법 1) OpenCV 함수, LUT 이용 변환
    #plt.imshow(table[img])          # 방법 2) Python의 인덱싱 생략 방법(for all value)
    plt.title('Gamma='+str(gamma));plt.xticks([]), plt.yticks([])
    i += 1
end_t = time.time()
print(f"execution time for Lab 1: {end_t-start_t:#.2f}[sec.]")
# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정


# 실습 2 ======================================================================================
# gamma > 1: 변환 함수를 LUT로 만들어 LUT 참조를 이용해 영상에 대해 감마 변환을 행한다.
#    방법: 지정된 감마에 대한 각 계조값에 대한 변환을 영상의 모든 영역에 대해 행한다.
#         매번 감마 함수를 호출하여 연산하기 때문에 시간이 더 많이 소요될 것으로 예상된다.
#         편의상 gamma < 1인 경우에 대해 행한다.#
# ============================================================================================
open_figure(win_name='Gamma Transformed Images using tedious computations: gamma < 1', fig_size=(12, 8), font_size=18)
plt.subplot(221), plt.imshow(imgC)      # 입력받은 대로 출력
plt.title('Original(color)')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])

img2 = img.copy()
print("img2.dtype=", img2.dtype)
print("img2.shape=", img2.shape)

ii = 2
start_t = time.time()               # 시간 측정을 위해 시작 시간 기록
for gamma in [0.6, 0.4, 0.2]:          # 밝게 만들기
#for gamma in [2, 3, 4]:             # 어둡게 만들기
    table = gammaTrans(gamma)
    plt.subplot(220+int(ii))

    # for loop를 이용하여 각 화소별로 명시적으로 지정
    # 가장 최악의 방법. 이렇게 loop를 수행하는 방식은 지양해야 함.
    row = img.shape[0]; col = img.shape[1]; ch = img.shape[2];
    rr = np.arange(0, row, 1); cc = np.arange(0, col,1); channel = np.arange(0, ch,1);
    #print("len(rr)=", len(rr), " len(cc)=", len(cc), " channel=", channel)
    for i in rr:
        for j in cc:
            for k in channel:
                val = img[i, j, k]
                img2[i, j, k] = table[val]
    plt.imshow(img2);

    plt.title('Gamma='+str(gamma))
    plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])
    ii += 1
end_t = time.time()
print(f"execution time for Lab 2: {end_t-start_t:#.2f}[sec.]")

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.show()
#plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
exit(0)
#=============================================================================================
