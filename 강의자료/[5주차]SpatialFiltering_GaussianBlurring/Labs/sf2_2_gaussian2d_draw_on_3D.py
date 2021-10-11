"""
개요
    1차원 가우시안 함수로부터 2차원 가우시안 커널을 만들고 이를 3차원 공간에 그려 보인다.
    2차원 가우시안 커널을 만들기위한 2차원 함수는
        방법 1) 1차원 함수로부터 가우스 2차 함수를 생성한다.
                이때, 정규화를 하지 않을 때나, 정규화를 할때 어떻게 다른지 관찰한다.
        방법 2) getGaussainKernel() 함수로 1차원 함수를 구한 다음 이것을 매트릭스 곱셈으로 2차원으로 확장하여 2차원 함수를 구한다.


"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# 넘파이 2차원 행렬을 행으로 줄을 나누어 정해준 형식으로 출력하는 함수
def prt_mtx(a, fmt):     # 어레이를 출력하는 함수
    print(f"\n  shape of the array={a.shape}")
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            print(f"{a[r, c]:{fmt}}", end="")
        print()

# ======================================================================================================================
# Draw 2D Gaussian functions on 3D space
# 2차원 Gaussian 함수의 3D 표현
# 2차원 가우시안 함수의 데이터를 반환하는 함수를 정의하고
# 이를 이용하여 다양한 시그마=2인 2차원 가우시안 함수를 3D 평면에 도시하시오. 높이가 크기가 됨.
# ======================================================================================================================

from mpl_toolkits.mplot3d import Axes3D

sigma = 3
sigma = 2

N = 7

#X = np.arange(-6, 6, 0.25)  # 범위=[-6, 6), +6 불포함. 참고용.
#Y = np.arange(-6, 6, 0.25)  # 범위=[-6, 6), +6 불포함. 참고용.
X = np.linspace(start=-N, stop=N, num=2*N+1, dtype=np.float32)       # X의 범위=[-N, +N]. stop를 포함하여 총 2*N+1개로 나눈다.
Y = np.linspace(start=-N, stop=N, num=2*N+1, dtype=np.float32)       # Y의 범위=[-N, +N]. stop를 포함하여 총 2*N+1개로 나눈다.

print(f"X.shape={X.shape}, Y.shape={Y.shape}")
XX, YY = np.meshgrid(X, Y)

def gaussian2D_no(x, y, sigma):       # 1) 정규화를 안했을 경우
    return(np.exp(-(x**2 + y**2)/(2*sigma**2)))

def gaussian2D_yes(x, y, sigma):        # 2) 정규화를 한 경우
    tmp = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    tmp = tmp/np.sum(tmp)
    return(tmp)

# 계수 계산법: 아래 1) 2) 중에 선택할 수 있다.
# 방법 1) 가우시안 함수로 부터 직접 계산한다. - 블러링에 활용하려면 정규화를 해야 한다.
#ZZ = gaussian2D_no(XX, YY, sigma)       # 정규화를 안한 경우. 합이 1보다 훨씬 크다.
ZZ = gaussian2D_yes(XX, YY, sigma)      # 정규화를 한 경우. 합이 1이다.


# 방법 2) OpenCV 함수로 부터 1D 커널을 구하여 이를 2차원으로 확장한다.
gauss1d = cv.getGaussianKernel(ksize=2*N+1, sigma=sigma, ktype=cv.CV_32F)       # 계수의 크기가 정규화된 커널을 반환한다.
ZZ = gauss1d @ gauss1d.T       # print(tmp.shape, np.sum(tmp)); ZZ = tmp/np.sum(tmp)  정규화 안해도 합은 1이다. 1차원부터 정규화서 들어왔음

print(f"ZZ.shape={ZZ.shape}, np.sum(ZZ)= {np.sum(ZZ)}")

fig = plt.figure(num='2D Gaussian function in 3D space')
# 아래는 다음의 경고오류를 출력함. 참고용
# MatplotlibDeprecationWarning: Axes3D(fig) adding itself to the figure is deprecated since 3.4.
# Pass the keyword argument auto_add_to_figure=False and use fig.add_axes(ax) to suppress this warning.
# The default value of auto_add_to_figure will change to False in mpl3.5 and True values will no longer work in 3.6.  This is consistent with other Axes classes.
#ax = Axes3D(fig)

ax = Axes3D(fig, auto_add_to_figure=False)
fig.add_axes(ax)

#ax.set_title("Gaussian: sigma=" + str(sigma))
ax.set_title("Gaussian: $\sigma$=" + str(sigma))
ax.plot_surface(XX, YY, ZZ, rstride=1, cstride=1, cmap='jet')





plt.show()










# 미션 1 --------------------------------------------------------------------------
# 위 소스를 참조로 하여 kx1 크기의 1차원 가우시간 커널을 반환하는 함수를 설계하시오
#       - 즉, getGaussianKernel() 함수를 설계하시오.
# 이를 기반으로 sepFilter2D() 함수를 사용하여 블러링을 행하시오.
# --------------------------------------------------------------------------------


# 미션 2 --------------------------------------------------------------------------
# 위 소스를 참조로 하여 kxk 크기의 2차원 가우시간 커널을 설계하시오.
# 위 소스를 참조로 하여 kxk 크기의 2차원 가우시간 커널의 3D 그래프를 출력하시오.
# kxk 2차원 커널을 이용하여 filter2D() 함수를 이용하여 영상의 블러링을 시행하시오.
# --------------------------------------------------------------------------------
