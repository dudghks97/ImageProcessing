"""
개요
    히스토그램 스트레칭과 감마 변환을 동시에 수행한다.
    입력의 범위와 출력의 범위를 각각 튜플로 지정한다.
    출력 범위의 값이 출력되도록 정해진 범위의 입력에 대해 감마함수 변환이 이루어집니다.
    => 이런 동작이 이루어지는 imadjust() 함수를 개발하여 수행하였다.
    이 함수는 mathworks사의 matlab 프로그래밍언어에서 지원하던 함수이다.
        mathworks 링크: https://ko.wikipedia.org/wiki/MATLAB
    matlab 언어는 수학, 공학자를 위한 파이썬과 유사한 프로그래밍 언어이다.
      imadjust() 함수 링크: https://www.mathworks.com/help/images/ref/imadjust.html?s_tid=srchtitle

    자체 제작한 imadjust() 함수를 이용하여 histogram stretching을 수행하였다.
    다야한 파라미터를 적용한 영상을 한 화면에 출력한다.

미션

1. 트랙바 기반 imadjust 영상 처리 수행프로그램을 제작하기
    1) 트랙바를 통해 imadjust 파라미터를 제어하시오.
    2) tkinter GUI를 이용해 처리 영상 파일을 선택할 수 있게 수정하시오.
    3) 소스 프로그램을 pyinstaller를 이용해 수행 파일로 만들고, 동작을 검증하시오.
        파라미터를 제어하여 대조비 감쇠, 증가, 감마변환, 역상 변환이 가능함을 보인다.

   위의 프로그램이 완성되면
    4) 변환된 영상에 대한 히스토그램을 imhist() 함수를 출력하는 기능을 추가하시오.
    5) tkinter GUI에 처리된 영상의 패스와 이름을 지정하는 기능을 추가하시오.
    6) pyinstaller를 통해 제작된 프로그램을 exe 파일로 저장하여 동작 가능함을 보이시오

2. Low contrast challenge
    https://www.dpreview.com/forums/post/64456273






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

# 중간에 계조값이 몰리는 영상
test_lst1 = [[(0, 1), (0, 1), 1], [(0.2, 0.8), (0, 1), 1], [(0.0, 0.8), (0, 1), 1.5], [(0.3, 0.7), (0, 1), 1.2]]

# 어두운 영상
test_lst2 = [[(0, 1), (0, 1), 1], [(0.0, 0.4), (0, 1), 0.6], [(0.0, 0.6), (0, 1), 0.7], [(0.0, 0.6), (0, 1), 0.5]]

Name = 'lenna.tif'
#Name = 'bone(m).bmp'
#Name = 'Fig0303(a)(breast).tif'
#Name = 'Fig0306(a)(bone-scan-GE).tif'
#Name = 'CarHeadLightGlare.jpg'
#Name = 'over_exposure.jpg' # input_range=(0.5, 1.0), output_range=(0, 1)
#Name = 'tiffany.bmp'      # input_range=(0.5, 1.0), output_range=(0, 1)
#Name = 'bk_gate.jpg'
#Name = 'bk_lobby.jpg'
#Name = 'dark1.png'
#Name = 'bukak.jpg'                 # 저조도
#Name = 'bone(m).bmp'
#Name = 'foggy.jpg'
#Name = 'low_contrast_challenge_mission.jpg'
#Name = 'unnamed.jpg'

Name = 'ele_book.jpg';  test_lst = test_lst1    # 대조비 약한 영상
Name = 'man_woman.bmp'; test_lst = test_lst2   # 어두운 영상

FullName = Path + Name
img = cv.imread(FullName)
#cv.imshow('src', img)
assert img is not None, "Failed to load image file:"

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

#=============================================================================================
# make_lut_for_imadjust() : imadjust() 함수를 만들기 위한 LUT를 만들어 반환한다.
# LUT에는 uint8형의 데이터에 대해 어떤 값으로 변환해서 출력해야 하는지를 담고 있는
# 256개의 uint8 형의 데이터가 저장되어 있다.
# =============================================================================================

def make_lut_for_imadjust(input_range=None, output_range=None, gamma=1):
    src = np.arange(0, 256) / 256
    if input_range is None:
        input_range = (0, 1)
    low_in, high_in = input_range
    #print(low_in, high_in)
    if output_range is None:
        output_range = (0, 1)
    low_out, high_out = output_range
    #print(low_out, high_out)
    dst = np.empty_like(src, np.float64)
    #print(dst.shape)

    for i in range(0, len(src)):
        if src[i] < low_in:
            dst[i] = low_out
        elif src[i] > high_in:
            dst[i] = high_out
        elif gamma == 1:
            dst[i] = (high_out-low_out) / (high_in - low_in) * (src[i] - low_in) + low_out
        else:   # gamma가 1이 아닌 경우 감마 변환 => 아래 루틴은 오류가 있습니다. (0, 1)=> (0, 1) 범위만 됩니다. 주석처리 하였음
            #dst[i] = ((src[i]-low_in) * (high_out-low_out)/(high_in - low_in))**gamma + (low_out-(1-high_in) * (1-high_in + src[i]) )
            dst[i] = (high_out - low_out) * ((src[i] - low_in)/ (high_in - low_in)) ** gamma + low_out      # 올바른 처리
    return src, dst, input_range, output_range, gamma    # dst가 LUT이다.

# =============================================================================================
# imadjust() 함수의 설계 사례
# 1. 코딩의 간편함을 위해 아래 가정을 전제하기로 합니다.
#   src는 uint8 영상
#   반환 영상도 uint8 영상
# 2. 먼저 LUT를 만들고, LUT를 기반으로 각 픽셀을 변환한다.
# =============================================================================================

# 소스 영상은 uint8 데이터형으로 가정한다.
def imadjust(img, input_range=None, output_range=None, gamma=1):
    _, LUT_float, __, ___, ____ = make_lut_for_imadjust(input_range, output_range, gamma)
    LUT = (255*LUT_float).astype(np.uint8)
    return LUT[img]


# -------------------------------------------------------------------------------------------
# imadjust() 적용 연습
# -------------------------------------------------------------------------------------------

#img 부동소수 쓰면 안된다. uint8 형만 입력 가능하다
#dst = imadjust(img, input_range=(0.01, 0.8), output_range=(0, 1), gamma=0.7)
#cv.imshow("test", dst); cv.waitKey(); exit()


open_figure(win_name='Histogram Stretching Results from self-made imadjust() function')

i = 0
for in_r, out_r, gm in test_lst:
    plt.subplot(241 + i)
    dst = imadjust(img[..., ::-1], input_range=in_r, output_range=out_r, gamma=gm)
    plt.title(fr'{i+1}) input={in_r}, out={out_r}, $\gamma$={gm}', fontsize=12)
    plt.imshow(dst)
    plt.axis("off")
    #plt.autoscale(enable=True,  # enable : True (default) turns autoscaling on,
    #              axis='both',  # axis : {'both', 'x', 'y'}, optional which axis to operate on
    #              tight=True)  # tight : bool or None, optional. If True, set view limits to data limits;
    plt.subplot(241 + i+1)
    dstG = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)
    plt.hist(dstG.ravel(), 256, [0, 256], histtype='step', color='c', label="gray")
    #plt.xticks(np.linspace(0, 255, 5))
    plt.autoscale(enable=True,  # enable : True (default) turns autoscaling on,
                  axis='x',  # axis : {'both', 'x', 'y'}, optional which axis to operate on
                  tight=True)  # tight : bool or None, optional. If True, set view limits to data limits;
    plt.grid(color='b', linestyle='--', linewidth=0.5)
    i += 2
# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정

plt.waitforbuttonpress()        # interactive mode에서 현 화면에서 버튼 입력을 기다린다.
#plt.show()







"""
# ---------------------------------------------------------------
# 아래는 매번 화소 값을 직접 변환하는 바람직하지 않는 접근 방식의 코딩 사례입니다....
# LUT 방식보다 속도가 매우 떨어지므로 지양해야할 방식의 코딩입니다.
# 기록 차원에서 남겨 둡니다. 오류가 있을 수도 있습니다....
# ---------------------------------------------------------------

def imadjust(src, input_range=None, output_range=None, gamma=1):
    if input_range is None:
        input_range = (0, 1)
    low_in, high_in = input_range
    print(low_in, high_in)
    if output_range is None:
        output_range = (0, 1)
    low_out, high_out = output_range
    print(low_out, high_out)
    dst = np.empty_like(src, np.float64)
    print(dst.shape)
    row, col = src.shape[0:2]
    if len(src.shape) == 3:         # mono?
        channel = 3
    else:
        channel = 1

    # 중요!!!! for loop를 사용하지 말고 구현할 수 있는지 검토해 보기 바랍니다...
    for i in range(0, row):
        for j in range(0, col):
            for k in range(0, channel):
                if src[i, j, k] < low_in:
                    dst[i, j, k] = low_out
                elif src[i, j, k] > high_in:
                    dst[i, j, k] = high_out
                else:
                    dst[i, j, k] = (high_out-low_out) / (high_in - low_in) * (src[i, j, k] - low_in) + low_out
    return dst

"""