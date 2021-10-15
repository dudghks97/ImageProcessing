"""
1. 개요
    2가지의 방법으로 가우시안 블러링을 구현한다.
    1) GaussainBlur() 함수 - 블러링을 위한 전용 함수
    2) 1차원 가우시안 함수로 만들어진 커널을 이용해 이를 가로 방향, 세로 방향에 대해 적용함으로써 2차원 가우시안 블러링을 행한다.
       sepFilter2D() 함수는 1차원 필터를 제공하여 x방향, y방향으로 각각 필터링을 행하는 함수이다.

2. 목표: 블러링을 OpenCV에서 지원하는 함수를 이용하여 구현하는 방법을 이해한다.
    실습 1) GaussianBlur() 함수를 사용하는 방법 - 원 영상에 1차원 혹은 2차원 가우시안 블러링을 행하는 함수
    실습 2) OpenCV의 1차원 가우시안 커널을 구하는 함수와 1차원 커널기반의 필터링함수, sepFilter2D의 사용법을 익힌다.
           이를 이용해 가로, 세로 각각 1회 필터링을 행한다.
           getGaussianKernel() - 1차원 가우시안 커널 구하기
           sepFilter2D() - 1차원 커널 2개로 x방향 1회, y 방향 1회 합하여 총 2회의 필터링을 행하기
           1차원 가우시안 필터로 2차원 평명에 대한 가우시안 블러링이 적용되는 원리를 이해한다.
3. 미션
    1. 실습 1의 예제를 pyplot으로 원본과 블러링 된 영상 2개(시그마=3, 시그마=15) 를 한 화면에 출력하시오.
    2. 원영상과 시그마(= 2, 5, 15)에 따른 다양한 출력 결과를 subplot을 이용하여 한꺼번에 출력하시오.
        조건 :
            시그마=2는 1차원 가우시안 커널로 가로 방향만 filter2D() 함수로 블러링. 필터는 가우시안 연산식을 통해 직접 정의할 것.
            시그마=5는 1차원 가우시안 커널로 세로 방향만 GaussianBlur() 함수로 블러링.
            시그마=15는 2차원 가우시안 커널로 양방향으로 filter2D() 함수로 블러링. getGaussianKernel을 기반으로 2차원 필터를 정의할 것.

4. 참고
    OpenCv-Python smoothing 부분
        https://docs.opencv.org/3.4.3/d4/d13/tutorial_py_filtering.html
    OpenCv-Python 한글 매뉴얼 - 기초 영상 처리
        https://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html



"""


import numpy as np
import cv2 as cv
# 만약 matplotlib 모듈이 없다면 윈도 컴맨드 창에서 "pip install matplotlib" 명령을 수행한다. 
from matplotlib import pyplot as plt



#----------  영상이 존재하는 폴더와 파일 이름을 지정하기.
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 data 폴더.
#Path = 'd:/work/StudyImages/images/'
#Name = 'monarch.bmp'
#Name = 'col_bar.png'
Name = 'lenna.tif'
Name = 'Mondrian.jpg'
Name = 'Resolution chart.tiff'

FullName = Path + Name

img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
cv.imshow(f'original', img)         # 원본 영상을 출력한다.


#"""
# ======================================================================================================================
# 실습 1 : GaussianBlur() 함수를 사용하는 방법
# GaussianBlur() 함수를 사용해 가장 편리한 방법으로 블러링한다.
#
# 	dst	= cv.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])
#       ksize=(가로, 세로). 예: (11, 1) 가로 방향 필터
#           ksize=(0, 0)으로 정하면 자동으로 적당한 크기를 계산하지만, sigma 값이면 0일 때는 오류가 발생한다.
#           대처 방안:
#               1) 블러링하지 않을 방향에는 0.1과 같은 작은 값의 시그마를 부여하면 이러한 오류를 회피할 수 있다.
#               2) 혹은 명시적으로 사이즈를 (1, N) 등으로 가로=1, 세로=N으로 핕터의 크기로 지정해 주어야 한다.
#                   필터의 크기를 지정할 때는 홀수로 지정하여야 한다.
#
# 미션 - 원본 영상과 파라미터에 따른 다양한 출력 결과를 subplot을 이용하여 한꺼번에 출력하시오.
#
# 질문 - 시그마 값에 비해 지나치게 작은 크기의 커널은 평균 필터와 같다. 이유를 설명하시오.
#
# ======================================================================================================================

sigma = 9

# ------------------------------------------------------------------------------------------------
# 활용 1) 양방향으로 블러링하는 사례
# 시그마를 제시하고 적절한 커널 크기를 자동 산정하게 하는 방안. 추천방안. 양방향 블러링
# ksize=(0, 0)-> 시그마로부터 계산. (가로, 세로)로 표기됨에 유의할 것. 대략; 커널 사이즈 = 6*sigma +1
# ksize=(0, 0)으로 지정하려면 sigma는 0이면 오동작한다. 0.1등으로 작은 값을 사용하면 동작함.
# ------------------------------------------------------------------------------------------------
blur = cv.GaussianBlur(img, (0, 0), sigma)     # 시그마=9의 값으로 양방향으로 블러링.
cv.imshow(f'1-1. both direction: (0, 0), sigma={sigma}', blur)

# 양방향으로 블러링. - 키워드를 제시하는 방식,. 순서 바꾸어도 된다.
blur = cv.GaussianBlur(src=img, ksize=(0, 0), sigmaY=sigma, sigmaX=sigma)   # 양방향 모두 블러링.
cv.imshow(f'1-2. both direction: (0, 0), sigmaY={sigma}, sigmaX={sigma}', blur)

# 양방향으로 블러링. - 키워드를 제시하는 방식,. sigmaX만 지정한다.
blur = cv.GaussianBlur(src=img, ksize=(0, 0), sigmaX=sigma)   # 주어진 커널이 2D이면 시그마를 X 방향 하나만 주어져도 양방향으로 블러링 한다.
cv.imshow(f'1-3. both direction: (0, 0), sigmaX={sigma}', blur)

# 양방향으로 블러링. - 키워드를 제시하는 방식,. sigmaX만 지정한다.
ks = (sigma*6+1, sigma*6+1)
blur = cv.GaussianBlur(src=img, ksize=ks, sigmaX=sigma)   # 주어진 커널이 2D이면 시그마를 X 방향 하나만 주어져도 양방향으로 블러링 한다.
cv.imshow(f'1-4. both direction: {ks}, sigmaX={sigma}', blur)

cv.waitKey()
cv.destroyAllWindows()


# ------------------------------------------------------------------------------------------------
# 활용 2) 한 쪽 방향으로만 블러링하는 사례
# sigmaX, sigmaY 값의 지정으로 X축 혹은 Y축 블러링을 선택적으로 시행할 수 있다.
# 1) 블러링하지 않을 방향에는 0.1과 같은 작은 값의 시그마를 부여하면 이러한 오류를 회피할 수 있다.
# 2) 혹은 명시적으로 사이즈를 (1, N) 등으로 가로=1, 세로=N으로 핕터의 크기로 지정해 주어야 한다.
# ------------------------------------------------------------------------------------------------

# x방향으로 블러링. - 키워드를 제시하는 방식
N = 6*sigma + 1          # 필터의 크기. 홀수여야 함.
#blur = cv.GaussianBlur(src=img, ksize=(0, 0), sigmaX=sigma, sigmaY=0.1) # 가로 방향만 블러링. 세로방향은 아주 약하게(0.1) 블러링.
blur = cv.GaussianBlur(src=img, ksize=(N, 1), sigmaX=sigma)        # (N, 1) =(가로, 세로)를 지정하면 괜찮다.
cv.imshow(f'2.1 x-direction: ({N}, 1), sigmaX={sigma}', blur)


# y방향으로 블러링. - 키워드를 제시하는 방식
#blur = cv.GaussianBlur(src=img, ksize=(0, 0), sigmaX=0, sigmaY=sigma)     # sigmaX=0이라서 오동작. 수행 중지.
#blur = cv.GaussianBlur(src=img, ksize=(0, 0), sigmaX=0.1, sigmaY=sigma)    # 세로 방향만 블러링. 가로방향은 아주 약하게(0.1) 블러링.
blur = cv.GaussianBlur(src=img, ksize=(1, N), sigmaX=0, sigmaY=sigma)     # y방향으로 블러링.
#blur = cv.GaussianBlur(src=img, ksize=(1, N), sigmaY=sigma)     # 안됨. 오동작. 수행 중지.
cv.imshow(f'2.2 y-direction: (1, {N}), sigmaX=0, sigmaY={sigma}', blur)
cv.waitKey()
cv.destroyAllWindows()


# ------------------------------------------------------------------------------------------------
# 활용 3) 커널 사이즈를 제시하고 그에 따라 자체적으로 적당한 시그마를 계산하는 방안
# ------------------------------------------------------------------------------------------------
N = 41
s = 0       # 시그마를 0으로 정하면 ksize에 따라 적절히 sigma를 산정한다.
ks = (N, N)     # ksize
blur = cv.GaussianBlur(img, ks, s)    # ksize=ks로 부터 시그마를 계산.
cv.imshow(f'3.1 automatic sigma generation: both direction: {ks}, sigma={s}', blur)

ks = (N, 1)     # ksize
blur = cv.GaussianBlur(img, ks, s)     # ksize로 부터 시그마를 계산. 가로 방향의 블러링
cv.imshow(f'3.2 automatic sigma generation: x-direction: {ks}, sigma={s}', blur)

ks = (1, N)     # ksize
blur = cv.GaussianBlur(img, ks, s)     # ksize로 부터 시그마를 계산. 세로 방향의 블러링
cv.imshow(f'3.3 automatic sigma generation: y-direction: {ks}, sigma={s}', blur)
cv.waitKey()
cv.destroyAllWindows()



# ------------------------------------------------------------------------------------------------
# 활용 4) 주의 사례: 연산 오류는 아니지만 바람직하지 않은 파라미터
# ------------------------------------------------------------------------------------------------
blur = cv.GaussianBlur(src=img, ksize=(51, 51), sigmaX=1)    # 시그마에 비해 커널의 크기가 크다. => 연산낭비. 손해 없다.
cv.imshow('4.1 too big kernel', blur)

blur = cv.GaussianBlur(img, (3, 3), 15)  # 시그마 값에 비해 지나치게 작은 크기의 커널을 사용한다. => 부정확 처리
cv.imshow('4.2 too small kernel', blur)

a = cv.getGaussianKernel(3, 15,  ktype=cv.CV_32F)
print(a)

cv.waitKey()
cv.destroyAllWindows()
exit(0)

#"""




#"""
# ======================================================================================================================
# 실습 2: OpenCV의 1차원 가우시안 커널을 구하는 함수와 1차원 커널기반의 코릴레이션 함수, sepFilter2D의 사용법을 익힌다.
#       getGaussianKernel - 1차원 가우시안 커널 구하기
#       sepFilter2D - 1차원 커널 2개로 이를 이용해 가로, 세로 각각 1회 총 2회의 필터링을 행한다.
# ======================================================================================================================


# -----------------------------------------------------------------------------------------------
# 1단계, 1번 창 출력 : 원본 영상을 출력한다.
# 평상 평면을 BGR -> RGB 순으로 재배열한다.
# matplotlib를 이용해 화면에 영상을 출력하려면 영상 평면 배열을 RGB로 바꾸어야 한다.
# OpenCV 함수를 이용해 BGR 평면 배열을 RGB 평면 배열로 바꾼다.
# -----------------------------------------------------------------------------------------------
#b, g, r = cv.split(img)   # img 파일을 b,g,r로 분리
#img = cv.merge([r, g, b]) # b, r을 바꿔서 Merge
img = img[..., ::-1]    # BGR -> RGB


# subplot(NMK)가 의미하는 것 : 화면을 rowXcolumn(NxM)개의 sub 창으로 나누고 K 번째 서브창을 지정한다.
# 이후 imshow() 함수는 지정된 sub 창에 출력한다.
plt.subplot(121)        # 2x2 서브 화면에서 1 번을 지정.
plt.imshow(img)
plt.title('Original')   # 지정된 sub 창에 그림 제목을 출력한다.
#plt.xticks([]), plt.yticks([])      # x축, y축에 label을 출력하지 않는다. 검은 테두리는 출력한다.
plt.axis('off') # # x축, y축에 label을 출력하지 않는다. 검은 테두리도 출력하지 않는다.


# -----------------------------------------------------------------------------------------------
# 2단계, 2번 창 출력 : 1차원 커널을 구하고 가로, 세로 각각 1회씩 1차원 필터링을 행한다.
# 가우시안 블러링 커널의 경우에는 위 결과가 2차원 필터를 사용하여 2차원 필터링을 행하는 것과 같다.
# 연산 시간은 1차원 필터링을 2번 행하는 것이 유리하다.
# -----------------------------------------------------------------------------------------------

# retval = cv.getGaussianKernel(ksize, sigma[, ktype] )
#   가우시안 커널을 반환하는 함수
#   The function computes and returns the ksize×1 matrix of Gaussian filter coefficients
#   ktype	Type of filter coefficients. It can be CV_32F or CV_64F

k = 31; s = 9; t = cv.CV_32F;
k_1D = cv.getGaussianKernel(ksize=k, sigma=s, ktype=t)
np.set_printoptions(precision=2)    # 소수 이하 2자리까지 출력. 이하 0이면 출력 안한다.
print('type(k_1D)=', type(k_1D), '| k_1D.shape=', k_1D.shape, '\nk_1D.T=\n', k_1D.T)


# 1차원 커널을 X축, Y축에 각각 1회씩 필터링(correlation) 연산을 수행한다.
# dst =	cv.sepFilter2D(src, ddepth, kernelX, kernelY[, dst[, anchor[, delta[, borderType]]]])
#   The function applies a separable linear filter to the image.
#   That is, first, every row of src is filtered with the 1D kernel kernelX.
#   Then, every column of the result is filtered with the 1D kernel kernelY.
#   The final result shifted by delta is stored in dst.

#dst = cv.sepFilter2D(src=img, ddepth=-1, kernelX=k_1D, kernelY=k_1D.T)     # 1) 커널을 ndarray로 지정
dst = cv.sepFilter2D(src=img, ddepth=-1, kernelX=k_1D, kernelY=k_1D)        # 2) Transpose 안시켜도 됨. 같은 결과
plt.subplot(122), plt.imshow(dst), plt.title('Gaussain Blurring: sigma=' + str(s)) # 1x2 서브 화면에서 2 번을 지정.
plt.axis('off')        # 라벨과 테두리 모두 없앤다.

# 이후 프로그램이 종료되므로 모든 창이 닫힌다.
plt.show()
exit(0)
#"""

