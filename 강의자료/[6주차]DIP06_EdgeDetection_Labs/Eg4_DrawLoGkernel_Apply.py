"""
1. 프로그램의 기능 - LoG(Laplacian of Gaussian) 함수를 만들고 그려 보인다.
    1) Gaussian 1차 미분 함수를 반환하는 함수를 정의하고 이를 3D로 표현한다.
    2) Gaussian 2차 미분 함수를 반환하는 함수를 정의하고 이를 3D로 표현한다.
    3) 위 함수의 커널을 이용하여 영상에 공간 필터링을 행한다.

2. 프로그램의 목표
    가우시안 1, 2차 미분 함수를 3차원 평면에 도시한다.
    함수를 커널화하여 활용하는 원리를 이해한다.

3. 프로그램에서 점검 포인트
    1) 함수 선언방법 -> gaussian 2D 함수의 1차, 2차 미분함수
    2) 함수의 커널화 작업
        -> 증분을 1로 정하고, 범위를 커널 크기 만큼 정한다.
        -> matplotlib.pyplot.meshgrid 함수를 이용해 x, y 좌표의 2차원 데이터를 생성한다.
    3) 미분 개념이 들어가기 시작하면 함수의 부호는 그다지 큰 의미가 없다.
        따라서 함수 자체에 -부호를 붙여 사용한 커널도 인터넷에 자주 등장한다.

4. 미션 - DoG filter
    DoG(Difference of Gaussian) filter란?
    2개의 서로 다른 시그마를 갖는 Gaussian Kernel(G1, G2)을 빼서 만든 필터이다. DoG = G1 - G2.
    1) 임의의 s1, s2의 시그마를 지정하면 DoG 필터와 커널의 크리 N을 반환하는 함수를 설계하시오.
    N, DoG = fspecial(filter='DoG', s1=2, s2=3)
    * fspecial() 함수는 matlab에서 제공하는 필터 설계 반환 함수입니다.
    2) 이 함수를 이용하여 영상에 DoG를 적용한 결과를 출력하시오.

5. 질문
    OpenCV에서 제공하는 LoG 관련 함수에 대하여 조사하고 이를 적용하여 분석한 실험 결과를 제시하시오.

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

#"""
# 실습 1 --------------------------------------------------------------------------
# 2차원 Gaussian 함수를 3D로 표현한다.
# ---------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

# 단계 1 - x, y 축의 범위를 커널에 상황에 맞추어 생성한다.
# 커널의 크기는 2N+1이 될 것이고, 증분은 1로 정한다.
# 대략적으로 시그마의 6배 정도를 커널의 크기로 정한다.
# 따라서 본 사례의 경우에는 NN으로 정하기로 한다.

sigma=3

NN = 2 * int(sigma * 3.5) +1 ; print('Kernel Size=', NN)
N = int(NN/2); print('N=', N)
X = np.arange(-N, N+1, 1); Y = np.arange(-N, N+1, 1)
print('length of X=', len(X)); print('length of Y=', len(Y));

# 단계 2 - x, y의 범위의 데이터를 2차원으로 확장한다.
XX, YY = np.meshgrid(X, Y)

# 단계 3 - 2차원 가우시안의 1차 미분 함수를 정의하고 위의 2차원 좌표 데이터를 대입하여 값을 생성한다.
# ZZ가 커널의 크기가 된다. 가우시안 블러 함수와는 달리 정규화하지 않는다.
def gaussian1stDev(x,y, sigma):
    r2 = x**2 + y**2
    return(np.exp(-r2/(2*sigma**2)))

ZZ = gaussian1stDev(XX,YY, sigma)
np.set_printoptions(precision=2, floatmode='fixed');    # 소수점 이하 2째 자리까지 출력.
print('ZZ.shape=', ZZ.shape);

# 아래 정규화과정이 없으면 코릴레이션 연산 값이 255를 넘어서 출력범위를 넘어 화면이 희게 출력된다.
acc = sum(sum(ZZ))      # 어레이의 모든 요소의 합을 구한다.
ZZ = ZZ/acc             # 정규화한다. 이로써 모든 요소의 합은 1이 된다.
if NN < 10:
    print(ZZ)

# 단계 4 - 3차원으로 도시한다.
fig = plt.figure(num='gaussian')
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("Gaussian: sigma=" + str(sigma))
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='jet')
ZZ_blur = ZZ.copy()
plt.show()
#plt.close('all')        # PyCharm에서 수행을 중지하면 모든 창이 닫힘.
#exit(0)




# 실습 2 --------------------------------------------------------------------------
# 2차원 Gaussian 1차 미분 함수를 3D로 표현한다.
# 단계 1, 2는 위의 정의된 것을 사용한다.
#  ---------------------------------------------------------------------------------
from mpl_toolkits.mplot3d import Axes3D

# 단계 1 - x, y 축의 범위를 커널에 상황에 맞추어 생성한다.
# 단계 2 - x, y의 범위의 데이터를 2차원으로 확장한다.

# 단계 1 - x, y 축의 범위를 커널에 상황에 맞추어 생성한다.
# 커널의 크기는 2N+1이 될 것이고, 증분은 1로 정한다.
# 대략적으로 시그마의 6배 정도를 커널의 크기로 정한다.
# 따라서 본 사례의 경우에는 NN으로 정하기로 한다.

sigma=3

NN = 2 * int(sigma * 3.5) +1 ; print('Kernel Size=', NN)
N = int(NN/2); print('N=', N)
X = np.arange(-N, N+1, 1); Y = np.arange(-N, N+1, 1)
print('length of X=', len(X)); print('length of Y=', len(Y));

# 단계 2 - x, y의 범위의 데이터를 2차원으로 확장한다.
XX, YY = np.meshgrid(X, Y)


# 단계 3 - 2차원 가우시안의 1차 미분 함수를 정의하고 위의 2차원 좌표 데이터를 대입하여 값을 생성한다.
# ZZ가 커널의 크기가 된다. 가우시안 블러 함수와는 달리 정규화하지 않는다.
def gaussian1stDev(x,y, sigma):
    r2 = x**2 + y**2
    r = np.sqrt(r2)
    return(-r/sigma**2 * np.exp(-r2/(2*sigma**2)))

ZZ = gaussian1stDev(XX,YY, sigma)
np.set_printoptions(precision=2, floatmode='fixed');    # 소수점 이하 2째 자리까지 출력.
print('ZZ.shape=', ZZ.shape);
if NN < 10:
    print(ZZ)


# 단계 4 - 3차원으로 도시한다.
fig = plt.figure(num='gaussian1stDev')
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("Gaussian 1st Derivative: sigma=" + str(sigma))
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='jet')
ZZ_1 = ZZ.copy()
plt.show()
#plt.close('all')        # PyCharm에서 수행을 중지하면 모든 창이 닫힘.




# 실습 3 --------------------------------------------------------------------------
# 2차원 Gaussian 2차 미분 함수를 3D로 표현한다.
# 단계 1, 2는 위의 정의된 것을 사용한다.
# ---------------------------------------------------------------------------------

# 단계 1 - x, y 축의 범위를 커널에 상황에 맞추어 생성한다.
# 커널의 크기는 2N+1이 될 것이고, 증분은 1로 정한다.
# 대략적으로 시그마의 6배 정도를 커널의 크기로 정한다.
# 따라서 본 사례의 경우에는 NN으로 정하기로 한다.

from mpl_toolkits.mplot3d import Axes3D
sigma = 3

NN = 2 * int(sigma * 3.5) +1 ; print('Kernel Size=', NN)
N = int(NN/2); print('N=', N)
X = np.arange(-N, N+1, 1); Y = np.arange(-N, N+1, 1)
print('length of X=', len(X)); print('length of Y=', len(Y));

# 단계 2 - x, y의 범위의 데이터를 2차원으로 확장한다.
XX, YY = np.meshgrid(X, Y)


# 단계 3 - 2차원 가우시안의 2차 미분 함수를 정의하고 위의 2차원 좌표 데이터를 대입하여 값을 생성한다.
# ZZ가 커널의 크기가 된다. 가우시안 블러 함수와는 달리 정규화하지 않는다.
def gaussian2ndDev(x,y, sigma):
    r2 = x**2 + y**2
    return( (r2-2*sigma**2)/sigma**4 * np.exp(-r2/(2*sigma**2)))

ZZ = gaussian2ndDev(XX,YY, sigma)
print('ZZ.shape=', ZZ.shape)
ZZ_2 = ZZ.copy()
if NN < 10:
    print(ZZ)

# 단계 4 - 3차원으로 도시한다.
fig = plt.figure(num='Lab 3: gaussian2ndDev')
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("Lab 3: Gaussian 2nd Derivative: sigma=" + str(sigma))
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='jet')
plt.show()


# 실습 4 --------------------------------------------------------------------------
# 2차원 Gaussian 2차 함수를 LoG(Laplacian of Gaussian) 커널이라 한다.
# LoG 커널을 이용하여 공간 필터링을 시행한다.
# ---------------------------------------------------------------------------------

#FullName = 'd:/work/@@DIP/LectureMaterials/Images/' + 'lenna.bmp'
FullName = '../data/' + 'lenna.tif'

img = cv.imread(FullName, 0)/255.0

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

# 단계 1 - x, y 축의 범위를 커널에 상황에 맞추어 생성한다.
# 커널의 크기는 2N+1이 될 것이고, 증분은 1로 정한다.
# 대략적으로 시그마의 6배 정도를 커널의 크기로 정한다.
from mpl_toolkits.mplot3d import Axes3D
sigma=3

NN = 2 * int(sigma * 3.5) + 1 ; print('Kernel Size=', NN)
N = int(NN/2); print('N=', N)

X = np.arange(-N, N+1, 1); Y = np.arange(-N, N+1, 1)
print('length of X=', len(X)); print('length of Y=', len(Y));

# 단계 2 - x, y의 범위의 데이터를 2차원으로 확장한다.
XX, YY = np.meshgrid(X, Y)


# 단계 3 - 2차원 가우시안의 2차 미분 함수를 정의하고 위의 2차원 좌표 데이터를 대입하여 값을 생성한다.
# ZZ가 커널의 크기가 된다. 가우시안 블러 함수와는 달리 정규화하지 않는다.
def gaussian2ndDev(x,y, sigma):
    r2 = x**2 + y**2
    return( ( (r2-2*sigma**2) * np.exp(-r2/(2*sigma**2)) ) / sigma**4)

# 참고: 가우스 2차 미분을 잘못 연산한 사례.
# 오류. r을 기반으로 2차 미분한 식.
def gaussian2ndDev_wrong(x,y, sigma):
    r2 = x**2 + y**2
    return( ( (r2-sigma**2) * np.exp(-r2/(2*sigma**2)) ) / sigma**4)


ZZ = gaussian2ndDev(XX,YY, sigma)
ZZ_wrong = gaussian2ndDev_wrong(XX,YY, sigma)



print('ZZ.shape=', ZZ.shape)
ZZ_3 = ZZ.copy()
np.set_printoptions(precision=2, floatmode='fixed');    # 소수점 이하 2째 자리까지 출력.
if NN < 10:
    print(ZZ)

# 단계 4a - 3차원으로 도시한다.
fig = plt.figure(num='Lab 4-step4a: gaussian2ndDev')
#ax = Axes3D(fig)
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("Lab 4-step4a: LoG- Gaussian 2nd Derivative: sigma=" + str(sigma))
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='jet')
plt.show()


# 단계 4b - 잘못된 커널을 3차원으로 도시한다.
fig = plt.figure(num='Lab 4-step4b: gaussian2ndDev_wrong')
ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)
ax.set_title("Lab 4-step4b: LoG wrong: Gaussian 2nd Derivative: sigma=" + str(sigma))
ax.plot_surface(XX, YY, ZZ_wrong, rstride=1, cstride=1, cmap='jet')
plt.show()

# 단계 5 - 영상에 LoG 커널을 사용해 공간 필터링을 행한다.
imgLoG = cv.filter2D(src=img, ddepth=-1, kernel=ZZ)
#imgLoG = cv.filter2D(src=img, ddepth=cv.CV_32F, kernel=ZZ)
print('imgLoG.dtype=', imgLoG.dtype)

# OpenCV로 출력한다. 비교 검토용.
#cv.imshow('org', img); cv.imshow('LoG', imgLoG); cv.waitKey(); exit(0)


plt.figure(num='Lab 4-step5: Filtered by LoG')
plt.subplot(121); plt.imshow(img, cmap='gray')
plt.title('Original'); plt.xticks([]), plt.yticks([])
plt.subplot(122); plt.imshow(imgLoG, cmap='gray')      # 3채널로 출력
plt.title('LoG image'); plt.xticks([]), plt.yticks([])


#plt.figure(num='Filtered by LoG')
#plt.subplot(121); plt.imshow(img, cmap='gray')
#plt.title('Original'); plt.xticks([]), plt.yticks([])
#imgLoG = cv.filter2D(src=img, ddepth=-1, kernel=ZZ_wrong)
#plt.subplot(122); plt.imshow(imgLoG, cmap='gray')      # 3채널로 출력
#plt.title('LoG image(wrong)'); plt.xticks([]), plt.yticks([])


plt.show()
plt.close('all')        # PyCharm에서 수행을 중지하면 모든 창이 닫힘.
#"""








"""
기록용: 정수화된 LoG 커널의 사례 
C_array = [
    [0, 0, 1, 1, 1, 0, 0],
    [0, 1, 3, 3, 3, 1, 0],
    [1, 3, 0,-7, 0, 3, 1],
    [1, 3,-7,-24,-7,3, 1],
    [1, 3, 0,-7, 0, 3, 1],
    [0, 1, 3, 3, 3, 1, 0],
    [0, 0, 1, 1, 1, 0, 0] ]

ZZ_2 = np.array(C_array, dtype=float )

"""