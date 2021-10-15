"""
참고 사이트
Histograms - 2: Histogram Equalization
https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html


1. 프로그램의 기능 및 점검 포인트 - 영상의 히스토그램 분포를 다음 4가지 방법으로 보인다.
    1) numpy의 지원함수(np.histogram)를 이용하여 히스토그램 데이터를 얻는다. 그림은 plot, bar로 그린다.
       이 함수는 ndarray로 선언된 2차원 데이터를 입력 받아 그 영상의 히스토그램 데이터를 반환한다.
       이 방법이 제일 단순한 듯하다.

    2) Python 지원 함수(matplotlib.pyplot.hist)를 이용하여 영상의 히스토그램을 그린다.
            https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html?highlight=matplotlib%20pyplot%20hist#matplotlib.pyplot.hist
        히스토그램 분포 데이터도 구할 수 있고 영상의 분포를 그림으로 표현하는 기능은 있으나 그림을 안그리는 제어 기능은 없다.
        이 hist()함수는 1차원 flattened array를 입력으로 사용해야 한다.
            1) 이를 위해 ravel() 함수/메소드를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.
                   https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
            2) 혹은 np.flatten() 함수를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.

    3) OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
        matplotlib.pyplot.plot 함수로 영상의 히스토그램을 그린다.
        a. 히스토그램의 원리 참고 자료
            https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
        b. OpenCV 히스토그램 함수, calcHist()의 사용법
        https://docs.opencv.org/3.4.3/d6/dc7/group__imgproc__hist.html#ga4b2b5fd75503ff9e6844cc4dcdaed35d

    4) OpenCV 지원 함수(cv.calcHist)를 이용하여 히스토그램을 만들고
        OpenCV함수(cv.line)로 히스토그램을 그리는 함수를 만들어 영상의 히스토그램을 그린다.
            OpenCV를 설치하면 다음 디렉터리에 예제 소스를 볼 수 있다.
            C:\opencv\sources\samples\python\hist.py

2. 프로그램의 목표
    1) 그래픽하게 표현된 히스토그램의 의미를 설명할 수 있다.
    2) matplotlib 혹은 OpenCV가 지원하는 기초적인 히스토그램 관련 함수의 운용법을 익힌다.


3. 미션

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



#=============================================================================================
# 단계 0
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#=============================================================================================
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
#Name = 'bk_lobby.jpg'


FullName = Path + Name
img = cv.imread(FullName)       # cv2 함수는 BGR 칼라 데이터를 읽는다.
assert img is not None, 'No image file....!'

# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾸고, GRAY 변환한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 이 부분을 활성화 시킨다.
# cv.imshow()를 사용하면 이 부분은 주석처리해야 한다.

b, g, r = cv.split(img); img = cv.merge([r, g, b]);
imgC = img.copy()      # img의 복사본을 하나 저장해 둔다. RGB 순으로 재 정렬되어 있다.
imgG = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
# interactive mode로 설정한다. plt.show() 없이 plt.imshow()만으로 출력된다.
# plt.ion();

#=============================================================================================

def open_figure(win_name='', fig_size=(17, 10), sub_title='', font_size=10):
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


#"""
#==============================================================================================
# 방법 2: Python 함수, matplotlib.pyplot.hist를 이용하여 영상의 히스토그램을 반환하고 그림을 그린다.
# 이 hist()함수는 1차원 flattened array를 입력으로 사용해야 한다.
#   1) 이를 위해 ravel() 함수/메소드를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.
#       https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html
#   2) 혹은 np.flatten() 함수를 이용해 2차원 영상 데이터를 1차원 flattened array로 변환한다.
# 입력 영상으로 1채널만 지원한다.

#
# ndarray.ravel() 함수 및 flatten() 메소드 연습
#       Return a contiguous flattened array
# >>> x = np.array([[1, 2, 3], [4, 5, 6]])
# >>> np.ravel(x)
# array([1, 2, 3, 4, 5, 6])
# >>> x.ravel()
# array([1, 2, 3, 4, 5, 6])
# >>> x.flatten()
# array([1, 2, 3, 4, 5, 6])
# >>> print(x.flatten())            # np.flatten(x)는 오류 발생.
# [1 2 3 4 5 6]
#==============================================================================================

# 원본 영상과 원본의 그레이 변환 영상을 출력한다.
plt.figure(num='Method 2: Original Images & their histograms: pyplot.hist()', figsize=(17, 10))
plt.suptitle("Method 2: by pyplot.hist()", fontsize=20, fontweight='bold')
# 창을 최대화 하여 그림을 출력한다.
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

plt.subplot(221); plt.imshow(imgC)      # 입력받은 대로 출력
plt.title('Original(color)')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])
plt.subplot(222); plt.imshow(imgG, cmap ='gray')
plt.title('Gray Image')
plt.axis('off')     #같은 동작 => plt.xticks([]), plt.yticks([])


# ---------------------------------------------------------------------------------------------
# 3) pyplot.hist() 함수로 gray 영상의 히스토그램을 출력한다.
# df, bin, patches = pyplot.hist()
# 이 함수는 히스토그램(DF, 분포함수) 혹은 누적 히스토그램(CDF)을 반환하면서도 히스토그램을 지정한 형식으로 그린다.
#       - 입력 파라미터 cumulative=True이면 누적분포 함수를 반환한다. default=False
# 그림을 안 그리게 하는 방법이 없어 보인다.
# ---------------------------------------------------------------------------------------------
plt.subplot(224);
plt.title("Gray Histogram: histtype='bar'")
plt.xticks([0, 255])
plt.grid(color='b', linestyle='--', linewidth=0.5)
# numpy.ravel(a, order='C'): Return a flattened array.
df, bin, patches = plt.hist(imgG.ravel(), 256, [0, 256], histtype='bar')     # 256, [0,256] => bin, range
#df, bin, patches = plt.hist(imgG.ravel(), 256, [0, 256], histtype='step');
print('df.shape=', df.shape, '| bin.shape=', bin.shape)
#print('type(patches)=', type(patches))      # 분석 보류: <class 'matplotlib.cbook.silent_list'>


# matplotlib.pyplot.hist(       # 파이썬의 히스토그램 함수. Plot a histogram
#   x,                      # x : (n,) array or sequence of (n,) arrays. 2차원 데이터는 ndarray.ravel()함수로 시퀀스 데이터로 펴야 함.
#   bins=None,              # bins : int or sequence or str, optional. 히스토그램을 담는 바구니.
#   range=None,             # 0~255의 계조치 범위.
#   density=None,
#   weights=None,
#   cumulative=False,       # 누적분포를 구하면 yes.
#   bottom=None,
#   histtype='bar',         #  {'bar', 'barstacked', 'step', 'stepfilled'},
#   align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None,
#   stacked=False, normed=None, *, data=None, **kwargs)[source]



# ---------------------------------------------------------------------------------------------
# 4) pyplot.hist() 함수로 color 영상의 히스토그램을 출력한다.
#   pyplot.hist() 함수를 3회 호출한다.
# ---------------------------------------------------------------------------------------------

plt.subplot(223)
plt.title("Color Histogram, histtype='step'")
plt.xticks([0, 255])
plt.grid(color='b', linestyle='--', linewidth=0.5)
i = 0
for ii in ['r', 'g', 'b']:
    plt.hist(img[:, :, i].ravel(), 256, [0, 256], histtype='step', color=ii, label=ii);
    i += 1

plt.legend(fontsize=10)
plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
#plt.show()
#exit(0)
#"""
