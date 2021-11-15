"""
1. 프로그램의 기능
    영상에 대해 2D FFT를 행하고 그 결과를 도시한다.
    실험 1 - numpy에서 제공하는 fft2() 함수로 영상에 대해 푸리에 변환을 행하고 그 결과를 보인다.
    실험 2 - cv에서 제공하는 DFT() 함수로 영상에 대해 푸리에 변환을 행하고 그 결과를 보인다.
            변환 결과는 numpy와 다르게 출력되는 것을 유의한다.

2. 프로그램의 목표
    2차원 푸리에 변환 데이터를 기반으로 파워 스펙트럼을 만들어 보일 수 있다.
    numpy에서 제공하는 푸리에 변환 관련 함수를 말하고 그 기능을 기술할 수 있다.

3. 프로그램에서 점검 포인트

4. 미션

5. 질문

참조 URL
    https://opencv-python.readthedocs.io/en/latest/doc/23.imageFourierTransform/imageFourierTransform.html
    참조 소스: o_09_03_FourierTransform2D.m


# Fourier Transform(푸리에 변환)
    . 시간 도메인(X축)에서 표현된 신호(일반적인 파형 도표)를 주파수 도메인으로 변환.
    . 시간축이 제거되어 대상의 전체적인 특징을 파악할 수 있음.
    . 이미지에 적용이 되어 중심이 저주파 영역, 주변이 고주파 영역을 나타냄.
    . 푸리에 변환을 하여 저주파 또는 고주파를 제거하고 다시 역으로 이미지로 변환 함으로써
      이미지가공을 할 수 있음.
      (ex; 푸리에 변환 후 중심의 저주파를 제거하고 다시 Image로 전환 하면 이미지의 경계선만 남게 됨.
           푸리에 변환 후 주변의 고주파를 제거하면 모아레 패턴(휴대폰으로 모니터를 찍었을 때 나타나는 현상)
           을 제거할 수 있음.(모니터의 고주파를 제거함.)
      )
"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt



#=============================================================================================
# 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'D:/StudyImages/images/'
#Path = 'data/'
#Name = 'lenna.tif'
#Name = 'monarch.bmp'
Name = 'fruits.jpg'
#Name = 'bone.jpg'p
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'
#Name = 'pen.bmp'
#Name = 'circle.bmp'

FullName = Path + Name

img = cv.imread(FullName)      # OpenCV에서는 BGR 배열임
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 블러링 시키면 스펙트럼이 어떻게 변화하는지 살핀다.
#img = cv.GaussianBlur(img, (0, 0), 5)

#======================================================================================================
# 실험 2 : OpenCV 푸리에 변환함수를 사용한다.
# cv.dft() 함수의 용법에 관한 주의 사항
#   1) 푸리에 변환/역변환 함수가 모두 cv.dft()를 사용할 수 있다. 역변환을 cv.idft() 함수를 사용할 수 있다.
#   2) 푸리에 변환 결과(복소수)는 2채널 ndarray로서 실수부와 허수부가 각각 0, 1 채널로 구성된다.
#   3) abs() 함수는 1채널의 복소수에 대해서는 1채널의 절대값을 반환하지만,
#       2채널로 구성된 복소수에 대한 abs() 함수는 2채널의 절대값을 반환한다.
#
# 영상에 대한 변환과 역변환 결과를 pyplot와 opencv 함수로 출력해 본다.
#======================================================================================================

def display(loc, img, title):
    plt.subplot(loc)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')

# 단계 0: 원본 영상 출력
#img = img/255
plt.figure(figsize=(15, 5))
plt.suptitle('OpenCV Fourier Transform: dft(), magnitude()', size=15)

plt.gray()
display(131, img, 'original')


# 단계 1: 푸리에 변환
f = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
print("dft: ", end=''), print(type(f), f.shape, f.dtype)

# 단계 2: 푸리에 역변환. 성공하였지만 역변환 결과는 복소수로 반환한다.
# 허수부가 0인 복소수를 반환한다.
# 2채널 복소 평면 데이터라 화면에 출력할 수 없다.
f_inv = cv.dft(f, flags=cv.DFT_INVERSE )
print("f_inv: ", end=''), print(type(f_inv), f_inv.shape, f_inv.dtype)
#plt.imshow(f_inv, cmap='gray')       # 오류 발생. 2채널 복소 평면

# 단계 3: 푸리에 역변환 2채널 데이터를 1 채널 실수 데이터로 변환한다.
mag = cv.magnitude(f_inv[:, :, 0], f_inv[:, :, 1])     # 복소수의 크기를 구한다. 1채널로 됨.
#mag = np.sqrt(f_inv[..., 0]**2 + f_inv[..., 1]**2)      # magnitude() 함수가 실제로는 이런 일을 한다.

print("mag: ", end=''), print(type(mag), 'shape=', mag.shape, 'type=', mag.dtype, 'min=', np.min(mag), 'max=', np.max(mag))
display(132, mag, 'inverse transform')

# 단계 4: 스펙트럼을 표현한다.
f_shift = np.fft.fftshift(f)
print("f_shift: ", end=''), print(type(f_shift), f_shift.shape, f_shift.dtype)
spectrum = 20*np.log10(cv.magnitude(f_shift[:, :, 0], f_shift[:, :, 1]) + 1)    # cv 지원 함수
#spectrum = 20*np.log(abs(f_shift)+1)        # python 지원 함수. 못 씀
print("spectrum: ", end=''), print(type(spectrum), spectrum.shape, spectrum.dtype)
display(133, spectrum, 'spectrum')

plt.show()

# OpenCV 창에 푸리엔 변환 경과 영상과 역변환 영상을 출략한다.
cv.imshow('spectrum', spectrum/np.max(spectrum))
#cv.imshow('inverse transform', mag)    # 255를 넘어서는 부동소수는 출력할 수 없다.
cv.imshow('inverse transform', mag/np.max(mag))
cv.waitKey()



# 정답
#cv.imshow('spectrum', spectrum/np.max(spectrum))
#cv.imshow('inverse transform', mag/np.max(mag))
