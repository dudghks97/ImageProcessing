"""
1. 개요
    getGaussainKernel()와 filter2D() 함수로 Gaussian Blurring을 행한다.
    1) 한쪽 방향(세로방향 혹은 가로 방향)의 블러링, 즉, 1차원 필터링을 행단다.
    2) 양쪽 방향으로, 즉,  2차원 필터링을 행한다.

2. 방법
    가우시안 커널을 만드는 함수: getGaussainKernel()
        - 이 함수는 (N, 1) 열벡터를 반환한다. 따라서 세로 방향으로 블러링을 행한다.
        - 가로 방향의 블러링이 필요하면 transpose를 취하여 (1, N) 벡터를 만든다.
        - 양방향 블러링을 위한 커널은 위의 열벡터화 행벡터에 대한 매트릭스 곱셉으로 구현단다. 죽, (N, 1) @ (1, N)
    filter2D() 함수를 통해 공간 필터링을 수행한다.
        - 이 함수에 적용 가능한 커널: (N, 1), (1, N), (N, N), (N, M)

3. 목표 - 숙지할 사항
    1) 2차원 공간필터의 생성과 2차원 공간필터링의 동작을 익힌다.
    2) 이를 이용해 가우시안 블러링이 적용되는 원리를 이해한다.
    3) getGaussainKernel()와 filter2D() 함수를 익힌다.

4. 함수 설명
    retval = cv.getGaussianKernel(ksize, sigma[, ktype] )
        The function computes and returns the ksize×1 matrix of Gaussian filter coefficients
        getGaussainKernel() 함수는 세로 방향의 블러링이 가능한 1차원 블러링을 위한 가우시안 커널을 반환한다. 즉, shape = (N, 1)
        numpy 관점에서는 ndim=2로서 2차원 커널이다.
        # 가로 방향으로 블러링하려면 transpose를 씌워 사용하면 된다.


5. 미션 -

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------
# 공통 섹션:  영상이 존재하는 폴더와 파일 이름을 지정하고 읽어내기
# ----------------------------------------------------------------------------------------------------------------------
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Path = 'd:/work/StudyImages/images/'
Name = 'lenna.tif'
#Name = 'col_bar.png'
#Name = 'monarch.bmp'
#Name = "Resolution chart.tiff"
#Name = 'Mondrian.jpg'

FullName = Path + Name

org = cv.imread(FullName)
#img = org[..., ::-1]/255        # matplotlib에서 출력을 위해, BGR -> RGB
img = org[..., ::-1]            # matplotlib에서 출력을 위해, BGR -> RGB

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.



#"""
# ======================================================================================================================
# 실습 1: 양방향 블러링
# ======================================================================================================================

fig = plt.figure(num="2D Gaussian Blurring with getGaussian/filter2D", figsize=(17, 10))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기
fig.patch.set_facecolor('green')       # 화면 바탕 색상을 정한다.
#fig.patch.set_facecolor('lightblue')       # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.1)        # Alpha, transparency: 0이면 투명, 1이면 완전 불투명.

plt.suptitle("2D Gaussian Blurring with getGaussian/filter2D", fontsize=14, fontweight='bold')

plt.subplot(2, 3, 1)
plt.imshow(img)
plt.axis("off")
plt.title("original")

knl_lst = []
sg_lst = [1, 3, 5, 7, 9]
for s in sg_lst:       # sigma list
    knl = cv.getGaussianKernel(ksize=6*s+1, sigma=s, ktype=cv.CV_32F)       # 1차원 가우시안 커널
    #print(knl.shape)        # (7, 1) ... (55, 1)
    knl2d = knl @ knl.T         # @는 매트릭스 곱하기 연산자. 2차원 가우시안 커널
    knl_lst.append(knl2d)


# enumerate() 함수는 ()안에 있는 sequence 자료를 나열하면서 그것을 인덱스 0, 1, 2,... 를 함께 반환한다.
# 예를 들어 enumerate("ABCD")는 for loop 안에서, (0, "A"), (1, "B"), (2, "C"), (3, "D")를 반환한다.
#
# zip() 함수는 내부에 2개의 같은 길이의 sequence 인자를 받아들여 이들을 순차적으로 쌍으로 반환한다.
# 예들들어 zip([5, 8, 3, 1], "ABCD")는 for loop 안에서 (5, "A"), (8, "B"), (3, "C"), (1, "D")를 반환한다.

for i, (knl, sigma) in enumerate(zip(knl_lst, sg_lst)):
    dst = cv.filter2D(img, -1, knl)
    plt.subplot(2, 3, i+2)
    plt.imshow(dst)
    plt.axis("off")
    plt.title(f"sigma={sigma}, shape={knl.shape}")

#plt.waitforbuttonpress()        # interactive mode. 'subplot'에서 버튼 입력을 기다린다.
#plt.show() exit(0)
#"""



# ======================================================================================================================
# 실습 2: 세로 혹은 가로 방향의 블러링
# ======================================================================================================================

fig = plt.figure(num="non interactive mode")
plt.suptitle("1 Dimensional Gaussian Blurring: getGaussian/filter2D", fontsize=14, fontweight='bold')

mng = plt.get_current_fig_manager()
#mng.window.state("zoomed")      # 현재의 창을 최대화한다.
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기
fig.patch.set_facecolor('silver')       # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.4)        # Alpha, transparency: 0이면 투명, 1이면 완전 불투명.

knl_lst = []
sg_lst = [1, 15, 25]

for s in sg_lst:       # sigma list
    knl = cv.getGaussianKernel(ksize=6*s+1, sigma=s, ktype=cv.CV_32F)       # 1차원 가우시안 커널
    print(f'1D Kernel of sigma={s}: shape={knl.shape}, dim={knl.ndim}')     # 1차원 커널이라 하지만, ndim=2으로 사실상 2차원 커널임.
    knl_lst.append(knl)

for i, (knl, sigma) in enumerate(zip(knl_lst, sg_lst)):
    dst = cv.filter2D(img, -1, knl)
    plt.subplot(2, 3, i+1)
    plt.imshow(dst)
    plt.axis("off")
    plt.title(f"sigma={sigma}, shape={knl.shape}")

for i, (knl, sigma) in enumerate(zip(knl_lst, sg_lst)):
    dst = cv.filter2D(img, -1, knl.T)
    plt.subplot(2, 3, i+4)
    plt.imshow(dst)
    plt.axis("off")
    plt.title(f"sigma={sigma}, shape={knl.T.shape}")

#plt.waitforbuttonpress()        # interactive mode. 'subplot'에서 버튼 입력을 기다린다.
plt.show()




