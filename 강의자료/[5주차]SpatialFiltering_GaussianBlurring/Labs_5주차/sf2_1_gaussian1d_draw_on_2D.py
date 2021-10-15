"""
1. 개요: 1치원 가우시안 함수를 만들고 이를 그려 보인다.
    실습 1: Draw 1D Gaussian functions on 2D space
        1차원 가우시안 함수의 데이터를 반환하는 함수를 정의하고
        이를 이용하여 다양한 시그마(표준편차, sigma)에 대한 함수 곡선을 색상을 바꾸어가며 그린다.
        이 함수 곡선은 최댓값이 1인것에 유념하자.

    실습 2.1: 1차원 가우시안 함수로부터 계산에 의해 1차원 가우시안 커널을 만들고 이를 그래프로 나타내 본다.
        블러링할 수 있는 가우시안 커널을 만들기 위해 정규화된 1차원 가우시안 함수의 데이터를 반환하는 함수를 정의하고
        이를 이용하여 다양한 시그마(표준편차, sigma)에 대한 함수 곡선을 색상을 바꾸어가며 그린다.
        정규화된 커널이라 함은 커널의 모두 계수의 합이 1이 되도록 스케일링하는 것이다.

    실습 2.2: getGaussainKernel() 함수로 정규화된 커널을 구해서 그림으로 출력하여 비교해 본다.
        이 함수는 정규화된 커널 값을 출력한다.



2. 목표 - 숙지할 사항
    실습 1에서는 pyplot을 이용한 함수의 drawing 기술에 대해 유념해야 한다.

3. 미션 - 소스 하단 부에 기술하였음.
    1) kx1 크기의 1차원 가우시간 커널을 반환하는 함수를 설계하시오
    2) kxk 크기의 2차원 가우시간 커널을 설계하시오.

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

#"""
# ======================================================================================================================
# 실습 1: Draw 1D Gaussian functions on 2D space
# 1차원 가우시안 함수의 데이터를 반환하는 함수를 정의하고
# 이를 이용하여 다양한 시그마(표준편차, sigma)에 대한 함수 곡선을 색상을 바꾸어가며 그린다.
# ======================================================================================================================

r = np.arange(-31, 31, 0.1, np.float32)     # r의 범위=[-31, =31)


def gaussian(r, sigma):     # 가우시안 함수를 정의함.
    return np.exp(-r**2/(2*sigma**2))

# curve_list =[ (시그마 값, 색상_스트링, 레전드_스트링), (...) , ...]
curve_list = [(1, 'b', r'$\sigma=1$'), (3, 'r', r'$\sigma$=3'), (5, 'c', r'$\sigma$=5'), (7, 'g', r'$\sigma$=7'), (9, 'm', r'$\sigma$=9')]
# 주의!!! 칼라 값을 아래와 같이 숫자로 지정하면 legend가 2배의 개수로  나타남.  이유 불명!!
# Writing mathematical expressions
#   https://matplotlib.org/tutorials/text/mathtext.html
#curve_list = [(1, 1, r'$\sigma=1$'), (3, 2, r'$\sigma=3$'), (5, 3, r'$\sigma=5$'), (7, 4, r'$\sigma=7$'), (9, 5, r'$\sigma=9$')]

for sigma, color, lbl in curve_list:
    plt.plot(r, gaussian(r, sigma), color, label=lbl)

#plt.ylim(top=1, bottom=0)       # 주의!!! 칼라 값을 숫자로 지정하면 y값의 범위가 늘어남. 이유 불명!!

plt.legend()
plt.grid(True)
plt.title('Gaussian functions with various sigma')
print("\n실습 2로 진행하려면 열려진 창을 닫아주세요.....")
plt.show()
#exit(0)
#"""




#"""
# ======================================================================================================================
# 실습 2.1: 1차원 가우시안 함수로부터 계산에 의해 1치원 가우시안 커널을 만들고 이를 그래프로 나타내 본다.
# 블러링할 수 있는 가우시안 커널을 만들기 위해 정규화된 1차원 가우시안 함수의 데이터를 반환하는 함수를 정의하고
# 이를 이용하여 다양한 시그마(표준편차, sigma)에 대한 함수 곡선을 색상을 바꾸어가며 그린다.
# 정규화된 커널이라 함은 커널의 모두 계수의 합이 1이 되도록 스케일링하는 것이다.
# ======================================================================================================================

r = np.linspace(start=-31, stop=31, num=63, dtype=np.float32)       # r의 범위=[-31, +31]

def n_gaussian(r, sigma): # 정규화된(normalized) 가우시안 함수를 정의함.
    y = np.exp(-r**2/(2*sigma**2))
    y /= np.sum(y)
    return y

fig = plt.figure(num="2.1) Normalized Gaussian Functions")
ax = fig.add_subplot(1, 1, 1)   # 서브 창을 1개만 연다.


# curve_list =[ (시그마 값, 색상_스트링, 레전드_스트링), (...) , ...]
curve_list = [(1, 'b', r'$\sigma=1$'), (3, 'r', r'$\sigma$=3'), (5, 'c', r'$\sigma$=5'), (7, 'g', r'$\sigma$=7'), (9, 'm', r'$\sigma$=9')]

# 주의!!! 칼라 값을 아래와 같이 숫자로 지정하면 legend가 2배의 개수로  나타남.  이유 불명!!
# Writing mathematical expressions
#   https://matplotlib.org/tutorials/text/mathtext.html
#curve_list = [(1, 1, r'$\sigma=1$'), (3, 2, r'$\sigma=3$'), (5, 3, r'$\sigma=5$'), (7, 4, r'$\sigma=7$'), (9, 5, r'$\sigma=9$')]

for sigma, color, lbl in curve_list:
    ax.plot(r, n_gaussian(r, sigma), color, label=lbl)

#plt.ylim(top=1, bottom=0)       # 주의!!! 칼라 값을 숫자로 지정하면 y값의 범위가 늘어남. 이유 불명!!
#ax.legend()
ax.legend(loc='center right', fontsize=15)   # fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}

plt.title('2.1) Gaussian functions with various sigmas', fontsize=20)
#ax.set_xlim([-31, 32])
#ax.set_ylim([0, 0.5])
# Major ticks & minor ticks in x-axis
major_ticks_x = np.arange(-31, +32, 2)
minor_ticks_x = np.arange(-31, +32, 1)
ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)


# Major ticks & minor ticks in y-axis
major_ticks_y = np.arange(0, 0.5, 0.05)
minor_ticks_y = np.arange(0, 0.5, 0.025)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# And a corresponding grid
ax.grid(which='both', linestyle='-', linewidth=1)
ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2)

#plt.tight_layout()      # 그림으로 꽉 채움.
plt.autoscale(enable=True, axis='both', tight=True)        # axis : {'both', 'x', 'y'}, optional


# ------------------------------------------------------------------------------------------------------------
# 실습 2.2: getGaussainKernel() 함수로 정규화된 커널을 구해서 그림으로 출력하여 비교해 본다.
# 이 함수는 정규화된 커널 값을 출력한다.
# retval = cv.getGaussianKernel(ksize, sigma[, ktype] )
# The function computes and returns the ksize×1 matrix of Gaussian filter coefficients
# ktype	Type of filter coefficients. It can be CV_32F or CV_64F
# ------------------------------------------------------------------------------------------------------------

N = 31
r = np.linspace(start=-N, stop=N, num=2*N+1, dtype=np.float32)       # r의 범위=[-31, +31]
fig = plt.figure(num="2.2) from getGaussianKernel()")
ax = fig.add_subplot(1, 1, 1)

#np.set_printoptions(precision=2)    # 소수 이하 2자리까지 출력. 이하 0이면 출력 안한다.
curve_list = [(1, 'b', r'$\sigma=1$'), (3, 'r', r'$\sigma$=3'), (5, 'c', r'$\sigma$=5'), (7, 'g', r'$\sigma$=7'), (9, 'm', r'$\sigma$=9')]
for s, color, lbl in curve_list:
    ax.plot(r, cv.getGaussianKernel(ksize=2*N+1, sigma=s, ktype=cv.CV_32F), color, label=lbl)

ax.legend(loc='center right', fontsize=15)   # fontsize : int or float or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
plt.title('2.2) getGaussianKernel with various sigmas', fontsize=20)

# Major ticks & minor ticks in x-axis
major_ticks_x = np.arange(-31, +32, 2)
minor_ticks_x = np.arange(-31, +32, 1)
ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)

# Major ticks & minor ticks in y-axis
major_ticks_y = np.arange(0, 0.5, 0.05)
minor_ticks_y = np.arange(0, 0.5, 0.025)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

# And a corresponding grid
ax.grid(which='both', linestyle='-', linewidth=1)
ax.grid(which='major', alpha=0.5)
ax.grid(which='minor', alpha=0.2)

#plt.tight_layout()      # 그림으로 꽉 채움.
plt.autoscale(enable=True, axis='both', tight=True)        # axis : {'both', 'x', 'y'}, optional

plt.show()
exit(0)
#"""



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
