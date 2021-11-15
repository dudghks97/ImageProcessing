"""
OpenCV 함수 기반의 푸리에 변환을 이용한 고주파 제거

1. 프로그램의 기능 - 고주파 성분을 제거하고 저주파 성분을 통과시킨다.
        통과 주파수대가 다른 점 외에 기능상으로는 FT3_filtering_numpy.py의 동작과 같음.
        1) 영상에 대해 2D FFT를 행하고 그 결과를 도시한다.
        2) 저주파 부분을 삭제하는 mask를 만든다.
        3) 이 마스크를 변환 결과에 곱하여 FFT 결과에서 지정한 부분(저주파 부분)을 아주 작은 값으로 만들어 제거한다.
        4) 변형된 변환 결과를 역변환하여 고주파 성분만을 가진 영상을 구한다.

2. 프로그램의 목표
    주파수 공간에서 특정 주파수를 통과 혹은 제거하는 과정을 기술할 수 있다.
    이런 목적을 위한 mask를 만들어 낼 수 있다.
    주파수 공간상에서 원하는 주파수 성분으로 영상으로 재구성할 수 있다.

3. 프로그램에서 점검 포인트
    주의사항!!:
        opencv로 푸리에 변환을 하면 복소수는 2채널로 반환한다. ch0=실수, ch1=허수
        따라서 mask도 2채널로 만들어야 한다.
    복소수의 크기를 구하려면 np.abs() 함수를 사용할 수 없다. magnitude() 함수를 사용해야 한다.

4. 미션
    1) 고주파가 제거된 영상에 ringing effect가 나타나는 이유가 무엇인가?
    2) FT3_filtering_draw_mask.py에서 제공하는 get_filter_mask() 함수를 활용하여 ringing effect가 제거된 필터링 결과를 구하시오.


참조 URL
    OpenCv dft() 함수 링크
        https://docs.opencv.org/4.1.2/d2/de8/group__core__array.html#gadd6cf9baf2b8b704a11b5f04aaf4f39d
    Discrete Fourier Transform
        https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
    https://opencv-python.readthedocs.io/en/latest/doc/23.imageFourierTransform/imageFourierTransform.html



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
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'D:/StudyImages/images/'
Path = 'data/'
Name = 'lenna.tif'
#Name = 'monarch.bmp'
#Name = 'fruits.jpg'
#Name = 'bone.jpg'p
#Name = 'woman_in_scarf(c).jpg'
#Name = 'man_woman.jpg'


FullName = Path + Name

# -------------------------------------------------------------------------------------------
# 단계 0: 원본 영상을 읽는다.
# -------------------------------------------------------------------------------------------

img = cv.imread(FullName)      # OpenCV에서는 BGR 배열임
# assert condition, message: condition이 false이면 message 출력하면서 AssertionError 발생.
assert img is not None, f"img={img}: 'No image file....[{Name}]!"

img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print('img.dtype=', img.dtype)

# -------------------------------------------------------------------------------------------
# 단계 1: OpenCV 함수로 퓨리에 변환을 행한다.
# -------------------------------------------------------------------------------------------
f = cv.dft(np.float32(img), flags = cv.DFT_COMPLEX_OUTPUT)

print("dft: ", end=''), print(type(f), f.shape, f.dtype)


# -------------------------------------------------------------------------------------------
# 단계 2: 스펙트럼을 출력한다.
# -------------------------------------------------------------------------------------------
plt.figure(num='Frequency Domain Filtering using OpenCV DFT', figsize=(10, 10))
#plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05, wspace=0.3)
#plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.1, hspace=0.1)
plt.suptitle('Frequency Domain Filtering using OpenCV DFT', size=15)

# 화면 1: 원본 영상 ---------
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.title('input')
plt.axis('off')

# 화면 2: 스펙트럼 영상 ---------
plt.subplot(222)
f_shift = np.fft.fftshift(f)        # OpenCV에는 fftshift() 함수가 없는 듯..
spectrum = 20*np.log(cv.magnitude(f_shift[:, :, 0], f_shift[:, :, 1]) + 1)    # cv 지원 함수
#spectrum = 20*np.log(abs(f_shift)+1)        # python 지원 함수. 못 씀
print("spectrum: ", end=''), print(type(spectrum), spectrum.shape, spectrum.dtype)
plt.imshow(spectrum, cmap='gray'), plt.title('spectrum'), plt.axis('off')


# -------------------------------------------------------------------------------------------
# 단계 3: 필터링 실시
# 푸리에 결과의 중앙부 저주파 영역만을 남기고 나머지 영역은 모두 0에 가까운 값으로 만들어 제거한다.
# 중앙에서 10X10 사이즈의 사각형의 값을 1로 설정함. 중앙의 저주파를 모두 제거
# 저주파를 제거하였기 때문에 배경이 사라지고 고주파(에지) 성분만 남게 됨.
# -------------------------------------------------------------------------------------------

# 함수 기반 마스크 정의
import FT3_filtering_draw_mask as me
rows, cols = img.shape[0], img.shape[1]

# 아래 선택 중의 하나를 호출하여 함수 호출로 필터를 설계한다.
mask_blr, mask = me.get_filter_mask((rows, cols), d=30, option='circle', sigma=5)   # 1)
#mask_blr, mask = me.get_filter_mask((rows, cols), d=30, option='square', sigma=5)   # 2)

print(f"mask.dtype={mask.dtype}, mask.shape={mask.shape}")
# 블러 필터를 쓰고, 저주파 통과 필터로 바꾼다.
mask = 1 - mask_blr
# OpenCV DFT 결과가 2채널로 생성되므로 마스크로 2채널로 확장한다.
mask = cv.merge([mask, mask])

# apply mask, 영상을 필터에 통과시킨다.
print(f"f_shift.shape={f_shift.shape}, mask.shape={mask.shape}")
fshift = f_shift * mask

# 화면 3: 스펙트럼 영상 ---------
plt.subplot(223)    # 변경된 스펙트럼
spectrum2 = 20*np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]) + 1)    # cv 지원 함수
#spectrum = 20*np.log(abs(fshift)+1)        # python 지원 함수. 못 씀
print("spectrum2: ", end=''), print(type(spectrum2), spectrum.shape, spectrum.dtype)
plt.imshow(spectrum2, cmap='gray'), plt.title('spectrum2'), plt.axis('off')

# -------------------------------------------------------------------------------------------
# 단계 4: 영상으로 역변환한다.
# -------------------------------------------------------------------------------------------
# 화면 4: 역변환 영상 ---------
f_ishift = np.fft.ifftshift(fshift)
img_flt = cv.idft(f_ishift)        # 역변환함수
img_flt = cv.magnitude(img_flt[:, :, 0], img_flt[:, :, 1])
plt.subplot(224)
plt.imshow(img_flt, cmap='gray')
plt.title('img_flt'), plt.axis('off')

plt.show()

