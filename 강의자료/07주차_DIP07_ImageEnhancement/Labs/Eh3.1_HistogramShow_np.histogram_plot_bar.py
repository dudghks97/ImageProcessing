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


#==============================================================================================
# 방법 1: np.histogram()를 사용하여 통계 데이터를 구하고 plot과 bar를 이용하여 히스토그램을 그린다.
# - 이 방법이 제일 쉬운 것 같다.
# 편리한 점 중 하나: np.histogram()는 입력 데이터를 1줄로 만들지 않아도 된다. 방법 2는 1줄로 만들어야 한다. 
#==============================================================================================

# numpy의 지원함수(np.histogram)를 이용하여 히스토그램 데이터를 얻는다.(그림을 그리지는 않는다)
# 이 함수는 ndarray로 선언된 2차원 데이터에 대한 히스토그램 데이터를 반환한다.
# https://docs.scipy.org/doc/numpy/ 에서 numpy.histogram 을 검색
# numpy.histogram(a, bins=10, range=None, normed=None, weights=None, density=None)
# return values:
#   hist : array. The values of the histogram. See density and weights for a description of the possible semantics.
#   bin_edges : array of dtype float. Return the bin edges (length(hist)+1).
#               hist의 길이보다 하나 많지만 맨 앞에 추가된 정보는 활용 가치가 없어 보인다.


# ---------------------------------------------------------------------------------------------
# 1) 히스토그램을 구하고, 그 정보를 출력하여 관찰한다.
# ---------------------------------------------------------------------------------------------
hist, bin_edges = np.histogram(imgG,    # gray 영상
                          256,          # bins : int or sequence of scalars or str, optional
                          [0, 255])     # range : (float, float), optional

# hist 각 화소 값들의 집산 개수. 같은 계조값에 따른 픽셀들의 갯수
# hist의 일부 데이터(맨 앞과 맨 뒷부분.)와 속성들을 출력해본다.
print('hist[50:80:10]=', hist[50:80:10])          # 화소값 50인 화소들의 갯수, 60인 화소들의 갯수, 70인 화소들의 갯수
print('hist[230:235]=', hist[230:235])        # 각 화소 값들의 집산 개수
print('hist.shape=', hist.shape)
print('len(hist)=', len(hist))

# bin_edges는 bins들의 레이블이라 할 수 있다.
# 0~255의 bin들에 대한 히스토그램을 구했으므로 bin_edges는 0~255의 가져야 한다.
# 그런데 실제로는 첫번째, 즉 bin_edges[0]는 0의 값을 가지고 있다.
print('\ntype(bins_edges)=', type(bin_edges))
print('bin_edges.dtype=', bin_edges.dtype)       # 화소 값. 왜 부동소수로 반환해 주는지 모르겠음. 왜 맨앞 하나를 추가하는지 알 수 없음.
bin_edges = bin_edges.astype(np.uint8)           # 출력의 편의를 위해 8비트 uint8 형으로 변환한다.
print('bin_edges.shape=', bin_edges.shape)
print('len(bin_edges)=', len(bin_edges))
# bin_edges의 일부 데이터(맨 앞과 맨 뒷부분.)를 출력해본다.
print('bin_edges[0:10]=', bin_edges[0:10])
#print('bin_edges[len(bin_edges)-10:]=', bin_edges[len(bin_edges)-10:])

# ---------------------------------------------------------------------------------------------
# 2) subplots() 함수와 autoscale() 함수로 화면을 둘로 나누고 화면에 꼭 차도록 그래프를 그린다.
# ---------------------------------------------------------------------------------------------

#f, (ax, ax2) = plt.subplots(nrows=2, num='histogram in bar & line: np.histogram')
# 그림(f)과 오브젝트(ax, ax2)를 반환

# subplots() 함수는 창의 이름과 sub 화면의 배열을 정의할 수 있다.
#       참고: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), num='Method 1: histogram in bar & line: np.histogram')

plt.suptitle("Method 1: by np.histogram()", fontsize=20, fontweight='bold')

# 원한다면 창을 최대화하여 그림을 출력한다.
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

ax1.imshow(imgG, cmap ='gray')      # gray 영상을 출력
# ax1.title() 지원 암함. 대신 text(x, y, s)가 있음. 원점에서 (x, y) 딸어진 곳에 s 스트링을 출력함.
x = 100         # (0,0) 그림의 좌측 최상단에서 오른 쪽으로 100 픽셀.
y = -10         # 위로 10 픽셀
ax1.text(x, y, 'Gray Image', fontsize=20)
ax1.axis('off')     # 테두리 못 그림 => ax1.xticks([]), ax1.yticks([])

ax2.imshow(imgC)        # color 영상을 출력
ax2.text(x, y, 'Original(color)', fontsize=20, color='r')
ax2.axis('off')

ax3.bar(bin_edges[1:], hist, width=1.0, label="histogram bar")  # 막대그래프로 그리기
ax3.autoscale(enable=True,       # enable : True (default) turns autoscaling on,
             axis='both',       # axis : {'both', 'x', 'y'}, optional which axis to operate on
             tight=True)        # tight : bool or None, optional. If True, set view limits to data limits;
ax3.legend()
ax3.grid(color='b', linestyle='--', linewidth=0.5)

for ii, v in enumerate(['r', 'g', 'b']):
    hist, bin_edges = np.histogram(imgC[..., ii],  # 칼라 영상의 i번째 채널
                                   256,  # bins : int or sequence of scalars or str, optional
                                   [0, 255])  # range : (float, float), optional
    ax4.plot(bin_edges[1:], hist, label=v, color=v)
#ax4.autoscale(enable=True, axis='both', tight=True)
ax4.autoscale(enable=True,  # enable : True (default) turns autoscaling on,
              axis='both',  # axis : {'both', 'x', 'y'}, optional which axis to operate on
              tight=True)  # tight : bool or None, optional. If True, set view limits to data limits;

ax4.legend()
ax4.grid(color='b', linestyle='--', linewidth=0.5)
#plt.show()
plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.

"""
# ---------------------------------------------------------------------------------------------
# 3) 히스토그램을 일부 손괴하여 그 결과를 bar 함수로 화면에 출력한다.
# 이때 그래프를 화면에 맞추어 마진이 없도록 조정한다. MATLAB 코드로 표현한다면 axis tight
# ---------------------------------------------------------------------------------------------
plt.figure(num='Spoiled Histogram: np.histogram')
hist2 = hist.copy()
hist2[150] = 0           # 실험적으로 일부 데이터를 0으로 만들어 그래프에 이 결과가 반영되는지 살핀다.
plt.axes(xlim=(0, 255), ylim=(0, np.max(hist)))     # 그래프의 경계 부분의 마진을 없앤다.
#plt.fill(bin_edges[1:], hist2)                      # 채우기 동작에 오류가 관찰됨. 원인 파악 못했음.
plt.bar(bin_edges[1:], hist2, width=1.0)             # bar로 그리면 문제 없음. width=바의 넓이
plt.xlabel('pixel value')
plt.grid(color='c', linestyle='--', linewidth=0.5)  # cyan색의 grid
plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.

#plt.show()
#exit(0)
"""

