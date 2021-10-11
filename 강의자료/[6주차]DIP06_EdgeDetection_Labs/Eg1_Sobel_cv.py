"""
****  다음 설명문은 Eg1_Sobel_cv.py + Eg1_Sobel_plt.py  두 예제 프로그램에 공통으로 적용된다. ***

1. 개요
    OpenCV 함수 Sobel를 이용하여 에지를 검출한 결과를 출력한다.
    pyplot.imshow()와 cv.imshow() 2가지 방법으로 결과를 출력해 본다.
    1) pyplot.imshow
        1채널, 부동소수 영상은 부호가 있는 수를 표현할 때 매우 편리하다.
            데이터의 범위나 자료형에 영향 받지 않고 최소, 최대를 구해 자체 정규화하여 출력한다.
        cmap 파라미터의 설정에 따라 여러 color map을 활용한 영상 출력을 시도할 수 있다.
        cv.imshow() 처럼 흑백 모노 그레이로 표현하려면 cmap='gray'로 설정하여야 한다.
        그렇지 않으면 default pseudo color map이 선택되어 칼라로 1채널 영상이 출력된다.
        - 개인적으로 따로 공부하기 바람.
    2) cv.imshow() 함수의 출력 범위
       부동소수 데이터: 0~1의 값만 출력한다. 음수는 0으로 간주하고 1을 넘는 값은 1로 간주한다.
       정수형 데이터: 0~255만 출력한다. signed int도 unsigned number로 간주한다.
       부동소수 혹은 unit8만 받아들인다.
       uint8이면 연산 중 발생한 8비트를 넘거나 음의 정수도 모두 하위 8비트만 취하여 출력한다.


2. 다음 내용을 숙지했는지 점검하기 바람
    1) cv.Sobel() 함수의 기본 사용법을 숙지한다.
    2) 에지 정보의 범위와 부호의 의미를 이해한다.
    3) 에지 정보를 선택적(부호, 범위)으로 화면에 pyplot.imshow()와 cv.imshow() 2가지 방법을 사용하여 출력할 수 있다.

3. 주요 함수
    dst = cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])
        src: input image.
        dst: output image of the same size and the same number of channels as src.
        ddepth:	output image depth, see combinations
        dx: order of the derivative x.
        dy:	order of the derivative y.
        scale: optional scale factor for the computed derivative values; by default, no scaling is applied (see getDerivKernels for details).
        delta: optional delta value that is added to the results prior to storing them in dst.
        borderType: pixel extrapolation method, see BorderTypes


4. 점검 사항
    실험 1의 질문에 답하시오.
    소벨 x, y축 에지 검출 결과를 합산해서 출력하여 보자.
    주의 사항 - 각 연산 결과에는 음수가 포함되어 있음. 절대 값을 붙이는 등의 절차가 필요함.

5. 미션

6. 참고
    출처 : https://docs.opencv.org/3.4.3/d5/d0f/tutorial_py_gradients.html


"""


import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
# np.clip() 함수 연습
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.clip.html
a = np.arange(10)   # [0 1 2 3 4 5 6 7 8 9]
print(a)
b = np.clip(a, 2, 7)
print(a)            # [0 1 2 3 4 5 6 7 8 9]
print(b)            # [2 2 2 3 4 5 6 7 7 7]
np.clip(a, 1, 6, out=b)
print(b)            # [1 1 2 3 4 5 6 6 6 6]
print(a)     # [0 1 2 3 4 5 6 7 8 9]
d = np.clip(a, [3,4,1,1,1,4,4,4,4,4], 8)
# a의 원소가 8보다 작을 때 앞의 list보다 작으면 리스트의 값. 그렇지 않으면 원래 원소 값.
# a의 원소가 8보다 크거나 같으면 8로 바꾼다. 
print(d)    # [3 4 2 3 4 5 6 7 8 8]
print(a)    # [0 1 2 3 4 5 6 7 8 9]
exit(0)
"""


# ----------------------------------------------------------------------------------------------------------------------
# 공통 부분 - 영상이 존재하는 폴더와 파일 이름을 지정하고 파일 읽기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'
#Path = 'd:/work/@@DIP/LectureMaterials/Images/'
#Name = 'monarch.bmp'
#Name = 'graphicShapes.png'
#Name = 'circles_rects_2.png'
Name = 'lenna.tif'
Name = 'circles_rects.png'


FullName = Path + Name

img = cv.imread(FullName, 0)                # 주의 !!! 모노 영상. 에지 영상은 그레이 영상에 대해 적용하는 것이 의미가 있음.
print("input image: type(img)=", type(img), " img.dtype=", img.dtype, "img.shape=", img.shape)

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.






#"""
# ======================================================================================================================
# OpenCV의 imshow()를 이용한 처리 결과 보이기
# cv.Sobel() 함수를 이용해 소벨 필터링을 행한다.
# 에지의 방향 및 부호를 관찰한다.
#   *** imshow() 함수의 출력 범위
#        부동소수 데이터: 0~1의 값만 출력한다. - 값은 0으로 간주하고 1을 넘는 값은 1로 간주한다.
#        정수형 데이터: 0~255만 출력한다. signed int도 unsigned number로 간주한다. 255를 넘는 값은 255로 간주한다.
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# 단계 1: 원본 영상을 보인다.
img = img/255       # 영상을 0~1의 부동소수로 만드는 것이 처리에 효울적이다.
print(f"step 1: img.dtype={img.dtype}, np.max(img)={np.max(img):#5.2f}, np.min(img)={np.min(img):#5.2f}")
cv.imshow('1. Original', img)
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# 단계 2: Sobel() 함수로 x방향의 소벨 미분을 행하고, 값의 범위를 관찰한 후 그 결과를 출력한다.
#   dst=cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]
#   출력결과의 데이터 타입은 음수를 포함할 수 있어야 한다. src가 uint8이라면 ddepth=-1은 안된다.
# 아래 1)~3)이 모두 x축 방향으로의 발생한 에지를 구한다.
# ----------------------------------------------------------------------------------------------------------------------

k = 3     # 커널 크기
# 1) 사용가능. x축 에지. dx=1, dy=0: x방향으로만 미분함을 의미한다.
# 출력결과는 CV_64F 부동소수이다.
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=k)

# 2) 사용가능. 위와 같은 동작. keyword를 지정하면 순서를 바꾸어도 상관 없다.
#sobelx = cv.Sobel(img, cv.CV_64F, dy=0, dx=1, ksize=k)

# 3) 사용가능. 아래는 ddept=-1이므로 img의 데이터 형에 따른다.
# 주의: 만약 img가 uint8 타입이라면 결과가 음수를 담을 수 없으므로 -1을 사용하면 안된다.
#       -1은 src와 같은 데이터 타입의 결과를 만들어 낸다.
#sobelx = cv.Sobel(img, -1, dy=0, dx=1, ksize=k)

# 소벨 값의 범위를 알아본다.
print(f'step 2.1 sobelx range: {np.min(sobelx):#.2f} < sobelx < {np.max(sobelx):#.2f}')

#sobelx /= np.max(sobelx)
cv.imshow('2a. sobel in x ', sobelx)
cv.waitKey()


# + 성분만 출력해 본다. 이 결과가 (2a)의 출력 결과와 같다.
# imshow()는 -성분은 값은 0으로 간주하고, +만 출력한다는 사실을 알 수 있다.
plus_compX = sobelx * (sobelx > 0)      # (sobelx > 0)는 논리 연산. 1과 0으로 된 어레이를 반환한다.
cv.imshow('2b. sobel in x, + only ', plus_compX)

# - 성분만 출력. imshow()는 +만 출력가능하므로 -성분을 추려 부호를 +로 바꾸어 출력함.
minus_compX = -sobelx * (sobelx < 0)
cv.imshow('2c. sobel in x, - only ', minus_compX)
cv.waitKey()

# 1) p_m: (+ 성분) + (- 성분을 +로 바꾼 성분)
# 2) np.abs(sobelx)는 sobelx의 음수와 양수를 모두 양수로 바꾸는 연산이다.
# 위 1)과 2)의 같은 값이라는 것을 그림과 원소값 비교로 확인해 본다.
p_m = plus_compX + minus_compX
cv.imshow('2d. sobel in x: (+) + (-)', p_m)
cv.imshow('2e. abs(sobel) in x ', np.abs(sobelx))
print('step 2.2: ', np.array_equal(p_m, np.abs(sobelx)))
# step 2.2:  True   # 두 어레이의 모든 원소의 값들이 각각 서로 같다.

cv.waitKey()
cv.destroyAllWindows()


# x축에 대한 분석과 중복되므로 출력은 가급적 생략한다.
# ----------------------------------------------------------------------------------------------------------------------
# 단계 3: y 방향에 대해 단계 2와 같은 동작을 수행한다.
# y방향의 소벨 미분을 행하고, 값의 범위를 관찰한 후 그 미분 결과를 출력한다.
# ----------------------------------------------------------------------------------------------------------------------
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=k)    # y축 에지. dx=0, dy=1: y방향으로만 미분함을 의미한다.

# 소벨 값의 범위를 알아본다.
print(f'step 3.1 sobely range: {np.min(sobely):#5.2f} < sobely < {np.max(sobely):#5.2f}')

#cv.imshow('3a. sobel in y ', sobely)
plus_compY = sobely * (sobely > 0)
cv.imshow('3b. sobel in y, + only ', plus_compY)     # + 성분만 출력

minus_compY = -sobely * (sobely < 0)
cv.imshow('3c. sobel in y, - only ', minus_compY)     # - 성분만 출력. -성분을 추려 부호를 +로 바꾸어 출력함.

p_m = plus_compY + minus_compY
cv.imshow('3d. sobel in y: |+| + |-| ', p_m)     # + 성분과 - 성분을 모두 출력
#cv.imshow('3e. abs(sobel) in y ', np.abs(sobely))
print('step 3.2: ', np.array_equal(p_m, np.abs(sobely)))        # step 3.2:  True
cv.waitKey()
cv.destroyAllWindows()



# ----------------------------------------------------------------------------------------------------------------------
# 단계 4: x, y방향의 소벨 미분을 합산한 결과를 출력한다.
# 질문:
#   아래 4a와 4b의 차이점은 무엇일까? 둘다 x축 미분과 y 축 미분을 더한 것인데...
# ----------------------------------------------------------------------------------------------------------------------
cv.imshow('4a. Wrong!: sobelx+sobely', sobelx+sobely)      # 잘못된 방법. x축, y축 부호가 다를 경우 상쇄된다.

abs_sobel_both = np.abs(sobelx)+np.abs(sobely)      # 올바른 방법
cv.imshow('4b. Good!: np.abs(sobelx)+np.abs(sobely)', abs_sobel_both)

# 최대 값이 1이 되도록 정규화. 해도 되지만 그다지 큰 의미는 없다.
# 일부로 강조해서 관찰하는 것이 필요한 경우도 있다.
n_sbl = abs_sobel_both / np.max(abs_sobel_both)
cv.imshow('4c. normalized absolute sobel_both', n_sbl)

#sobelxy = cv.Sobel(img, cv.CV_64F, dx=1, dy=1, ksize=1)     # 이것은 2차 미분 필터를 사용한다.
#cv.imshow('4c. sobelxy', 3* np.abs(sobelxy))
cv.waitKey()
cv.destroyAllWindows()


"""
# 아래 기능은 plt.show()가 내부에서 자동으로 처리해 주기 때문에 생략하기로 한다.
# 여기서는 CV에서 plt.show()의 자동 정규화를 흉내본 것이다.
# ----------------------------------------------------------------------------------------------------------------------
# 단계 5: 소벨값 0을 0.5로.. -최소값은 0으로 +최대값은 1로 정규화를 행한다.
# 즉, 에지 성분이 없으면 128의 화소 값을 갖고, 음수의 에지값은 0~127까지, 양수의 에지값은 129~255까지의 범위를 갖게 표현해 본다.
# 어두울 수로 -값이 강한 에지이고, 밝을 수록 +에지 값이 강한 것으로 이해할 수 있다.
# ----------------------------------------------------------------------------------------------------------------------

# 1) x축 방향 에지 관찰

# 정규화. 0.5 ~ 1.0
nor_plus_compX = 0.5 * plus_compX / np.max(plus_compX)

# 정규화. 0~0.5
nor_minus_compX = 0.5 - 0.5 * minus_compX / np.max(minus_compX)
cv.imshow('5.1 sobelx in half tone', nor_plus_compX + nor_minus_compX)

# + 성분만 출력
cv.imshow('2b. sobel in x, + only ', plus_compX)

# - 성분만 출력. -성분을 추려 부호를 +로 바꾸어 출력함.
cv.imshow('2c. sobel in x, - only ', minus_compX)

cv.waitKey()
cv.destroyAllWindows()

# 2) y축 방향 에지 관찰
nor_plus_compY = 0.5 * plus_compY / np.max(plus_compY)     # 정규화. 0.5 ~ 1.0
nor_minus_compY = 0.5 - 0.5 * minus_compY / np.max(minus_compY)     # 정규화. 0~0.5
cv.imshow('5.2 sobely in half tone', nor_plus_compY + nor_minus_compY)
cv.imshow('3b. sobel in y, + only ', plus_compY)     # + 성분만 출력
cv.imshow('3c. sobel in y, - only ', minus_compY)     # - 성분만 출력. -성분을 추려 부호를 +로 바꾸어 출력함.

cv.waitKey()
exit(0)
"""


"""
# 기록용: 분석할 필요 없음....
# 아래의 파일 비교에 의한 두 어레이의 영상 내용 비교가 적절치 않은 이유를 설명하시오...??
def compare2array():
    cv.imwrite('2a.png', sobelx)
    cv.imwrite('2b.png', np.abs(sobelx))
    img_2a = cv.imread('2a.png')
    img_2b = cv.imread('2b.png')
    print(np.max(img_2a), np.max(img_2b))       # 이런 값이 나온 배경을 추론하여 설명하시오.
    print(np.min(img_2a), np.min(img_2b))
    cv.imshow("difference", 100 * np.abs(img_2a-img_2b))
    print(np.array_equal(img_2a, img_2b ))          # 다름. false
#compare2array(); cv.waitKey(); exit(0)


"""