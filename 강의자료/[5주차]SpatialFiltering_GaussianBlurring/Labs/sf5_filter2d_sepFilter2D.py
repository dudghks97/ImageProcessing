"""
개요
    filter2D() 함수와 sepFilter2D()의 연산 시간을 비교한다.
      sepFilter2D() 함수에는 2차원 필터링을 하는데도 1차원 필터를 2개 제공한다.
	  가로로 처리가 끝난 결과로 다시 세로필터를 써서 그 결과를 반납한다.

설명
    각 함수의 연산 회수는 sepFilter2D()가 2*K번이라면 filter2D()는 KxK번이다.
    연산회수 관점에서는 sepFilter2D가 매우 유리할 것으로 보인다.
    실제로 실험을 해보면 예상외로 filter2D() 함수가 sepFilter2D() 보다 수행 속도가 더 빠르다.

이유
    filter2D() 함수는 11x11 이상이면 DFT 변환에 의해 연산을 수행한다고 기술되어 있다.
    DFT 변환과 역변환에 시간이 소요되지만 일단 변환이 되면 코릴레이션 연산은 두 어레이의 요소간 곱셈으로 단순화되기 때문에
    아무리 커널이 커도 시간이 별로 더이상 증가되지 않는 것이다.
    큰 필터를 사용할 수록 DFT 변환을 사용하는 filter2D() 함수가 유리해질 것으로 예상된다.
    sepFilter2D()가 유리하다는 발상은 과거 SIMD 구조, GPU 등의 가속기가 지원되지 않는 환경에서 적용되는 분석이라고 추정된다.

    *** 맨 아래 주석문 참고

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
#Name = 'Resolution chart.tiff'
Name = 'col_bar.png'
Name = 'lenna.tif'

FullName = Path + Name

img = cv.imread(FullName)
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
#cv.imshow(f'original', img)         # 원본 영상을 출력한다.

# --------------------------------------------------------------------------------------------
# 실험 1 : 선형 분리가능 필터링의 실험 결과를 출력한다.
# --------------------------------------------------------------------------------------------

sg = 5  # 시그마
k1 = cv.getGaussianKernel(ksize=6*sg+1, sigma=sg, ktype=cv.CV_32F)
dst = cv.sepFilter2D(src=img, ddepth=-1, kernelX=k1, kernelY=k1)    #  kernelY=k1.T도 같은 결과
cv.imshow(f"sepFilter2D: s={sg}, k1.shape={k1.shape}", dst)
cv.waitKey()
exit()






# --------------------------------------------------------------------------------------------
# 실험 2 : 선형 분리가능 필터링과 일반 2차원 필터링의 수행시간을 테스트해 본다.
# --------------------------------------------------------------------------------------------




# ======================================================================================================================
# 가우시안 1차원 필터와 2차원 필터의 선언
# ======================================================================================================================

# 아래 데이터 세트 중의 하나를 선택한다. 3종의 시그마와 필터 크기를 대해 실험할 예정
# 1) data set 1: big sigma and filter size
k_lst = [21, 51, 91]        # list of kernel size
sg_lst = [5, 9, 15]         # list of sigma

# 2) data set 2: small sigma and filter size
#k_lst = [5, 7, 9]        # list of kernel size
#sg_lst = [1, 1.5, 2]         # list of sigma


k1_lst = []     # 1d kernel list
k2_lst = []     # 2d kernel list
t = cv.CV_32F

for ksz, sg in zip(k_lst, sg_lst):
    k1 = cv.getGaussianKernel(ksize=ksz, sigma=sg, ktype=t)
    k2 = k1 @ k1.T
    k1_lst.append(k1)
    k2_lst.append(k2)

#np.set_printoptions(precision=2)    # 소수 이하 2자리까지 출력. 이하 0이면 출력 안한다.
#print('type(k_1D)=', type(k_1D), '| k_1D.shape=', k_1D.shape, 'sum(k_1D)=', np.sum(k_1D), '\nk_1D.T=\n', k_1D.T)


import time

iter = 1000
iter = 10       # 연습용은 작은 값을 사용하세요....
print("iteration =", iter)

# ======================================================================================================================
# filter2D() 함수의 수행 시간 측정
# 2d kernel list, k2_lst를 사용한다.
# ======================================================================================================================

print("\nfilter2D() 함수의 수행 시간 측정")
fil_recrd_k2 = []           #   filter2D() 함수의 수행 기록
for k2 in k2_lst:
    s = 0
    start = time.time()
    for i in range(iter):
        dst = cv.filter2D(img, -1, k2)
        s += np.sum(dst)
    end = time.time()
    print(f'kenel shape={k2.shape}, sum={s}')
    elapsed = end-start
    fil_recrd_k2.append(elapsed)
    print('filter2D: elapsed time=', elapsed)



# ======================================================================================================================
# sepFilter2D() 함수의 수행 시간 측정
# 1d kernel list, k1_lst를 사용한다.
# ======================================================================================================================
print("\nsepFilter2D() 함수의 수행 시간 측정")
sepF_recrd_k1 = []           #   sepFilter2D() 함수의 수행 기록
for k1 in k1_lst:
    s = 0
    start = time.time()
    for i in range(iter):
        dst = cv.sepFilter2D(src=img, ddepth=-1, kernelX=k1, kernelY=k1)
        s += np.sum(dst)
    end = time.time()
    print(f'kenel length={len(k1)}, sum={s}')
    elapsed = end-start
    sepF_recrd_k1.append(elapsed)
    print('sepFilter2D: elapsed time=', elapsed)





# ======================================================================================================================
# 실험 결과 - 실험 데이터 세트 1(작은 필터)
# 예상외로 filter2D() 함수가 sepFilter2D() 보다 수행 속도가 더 빠르다.
# 연산 회수는 sepFilter2D()가 2*K번이라면 filter2D()는 KxK번이다.
# 이는 과거 SIMD 구조, GPU 등의 가속기가 지원되지 않는 환경에서 적용되는 분석이라고 추정된다.
# sepFilter2D = 2.1/2.6/3.1
# filter2D = 1.1/2.0/3.0
# ======================================================================================================================

#k_lst = [5, 7, 9]        # list of kernel size
#sg_lst = [1, 2, 3]         # list of sigma
# iteration = 1000
#
# len(k1)=5, sum of sepFilter2D: 94414595000
# sepFilter2D: elapsed time= 2.1272289752960205
#
# len(k1)=7, sum of sepFilter2D: 94414531000
# sepFilter2D: elapsed time= 2.6906347274780273
#
# len(k1)=9, sum of sepFilter2D: 94414331000
# sepFilter2D: elapsed time= 3.142857789993286
#
#
# k2.shape=(5, 5), sum of filter2D: 94414595000
# filter2D: elapsed time= 1.1884477138519287
#
# k2.shape=(7, 7), sum of filter2D: 94414531000
# filter2D: elapsed time= 1.9864752292633057
#
# k2.shape=(9, 9), sum of filter2D: 94414331000
# filter2D: elapsed time= 3.0344364643096924



# ======================================================================================================================
# 실험 결과 - 실험 데이터 세트 2(큰 필터)
# 계산량의 비교(2K vs KxK)와는 반대로 오히려 filter2D가 훨씬 빠르다.
# 이는 고속화된 DFT 연산을 이용한 계산량의 절감때문인 것으로 분석된다.
#
# 큰 필터에 대해서는 필터가 더 커져도 filter2D 함수는 시간 소요가 크게 증가하지 않았다.
# 예상한 대로 필터의 크기에 영향 받지 않는 DFT 연산이 활용되었기 때문으로 추정할 수 있다.
# sepFilter2D = 6.8/16.1/29.8
# filter2D = 6.7/8.7/9.9
# ======================================================================================================================

# k_lst = [21, 51, 91]        # list of kernel size
# sg_lst = [5, 9, 15]         # list of sigma
#
# iteration = 1000
#
# len(k1)=21, sum of sepFilter2D: 94414083000
# sepFilter2D: elapsed time= 6.804197072982788
#
# len(k1)=51, sum of sepFilter2D: 94416524000
# sepFilter2D: elapsed time= 16.106767177581787
#
# len(k1)=91, sum of sepFilter2D: 94414006000
# sepFilter2D: elapsed time= 29.79546356201172
#
#
# k2.shape=(21, 21), sum of filter2D: 94414083000
# filter2D: elapsed time= 6.709491729736328
#
# k2.shape=(51, 51), sum of filter2D: 94416521000
# filter2D: elapsed time= 8.68143367767334
#
# k2.shape=(91, 91), sum of filter2D: 94414011000
# filter2D: elapsed time= 9.913622617721558
