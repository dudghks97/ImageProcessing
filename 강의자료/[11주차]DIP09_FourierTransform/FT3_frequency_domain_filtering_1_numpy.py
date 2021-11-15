"""

제목: Numpy 함수를 이용한 고주파 통과 필터링

이 프로그램은 import FT3_filtering_draw_mask 프로그램을 모듈로 활용합니다.

1. 프로그램의 기능 - 저주파 성분을 제거하고 고주파 성분을 통과시킨다.
        1) 영상에 대해 2D FFT를 행하고 그 결과를 도시한다.
        2) 저주파 부분을 삭제하는 mask를 만든다.
            - 이때 FT_filtering_draw_mask.py 모듈의 get_filter_mask() 함수를 활용한다.
        3) 이 마스크를 변환 결과에 곱하여 FFT 결과에서 지정한 부분(저주파 부분)을 아주 작은 값으로 만들어 제거한다.
        4) 변형된 변환 결과를 역변환하여 고주파 성분만을 가진 영상을 구한다.

2. 프로그램의 목표
    주파수 공간에서 특정 주파수를 통과 혹은 제거하는 과정을 기술할 수 있다.
    이런 목적을 위한 mask를 만들어 낼 수 있다.
    주파수 공간상에서 원하는 주파수 성분으로 영상으로 재구성할 수 있다.

3. 프로그램에서 점검 포인트
    fftshift(), iffttshift() 역할을 이해한다.
    역변환 최종 과정에서 왜 ifftshift() 결과를 사용하는지를 이해한다.

4. 미션
    저주파 부분만 통과시키도록 프로그램을 수정하시오.

5. 질문

참조 URL
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
# 단계 1: 푸리에 변환을 수행하고 그 스펙트럼을 보인다.
# -------------------------------------------------------------------------------------------
f = np.fft.fft2(img)    

plt.figure(num='Spectrums & filtering', figsize=(16, 8))
#plt.subplots_adjust(hspace=0.6, top=0.95, bottom=0.05, wspace=0.3)
plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.1)

spectrum = 20*np.log10(np.abs(f)+1)
plt.subplot(241)
plt.imshow(spectrum, cmap='gray')
plt.title('(a) spectrum: f=fft2(img)', fontsize=15)
plt.axis('off')


# -------------------------------------------------------------------------------------------
# 단계2 : 푸리에 변환 결과를 shift하고 그 스펙트럼을 보인다.
# shift하는 이유: 주파수가 0인 부분(DC 성분)을 중앙(원점)에 위치하게 하여 분석을 용이하게 하기 위함.
# 이로서 중앙에 저주파가 모이게 된다.
# -------------------------------------------------------------------------------------------
fshift = np.fft.fftshift(f) 

spectrum = 20*np.log10(np.abs(fshift)+1)
plt.subplot(245)
plt.imshow(spectrum, cmap='gray')
plt.title('(e) spectrum after shift: fftshift(f)', fontsize=15)
plt.axis('off')


# -------------------------------------------------------------------------------------------
# 단계 3: 통과할 주파수 영역을 지정하는 마스크(mask)를 제작한다.
# 마스크는 spectrum after shift 의 크기와 같은 0과 1로 이루어진 행렬이다.
# 0인 값은 해당 지점의 주파수를 제거하는 역할을 하며, 1인 값은 해당 지점의 주파수를 통과시키는 역할을 한다.
# 현재의 프로그램은 원점 부분의 주파수를 제거하는 마스크를 만들었다.
# 이는 푸리에 변환 결과의 중앙부 저주파 영역을 0에 가까운 값으로 만들어 저주파 성분을 제거하게 될 것이다.
# 저주파를 제거하였기 때문에 배경이 사라지고 고주파(에지) 성분만 남게 된다.
# -------------------------------------------------------------------------------------------

import FT3_filtering_draw_mask as me

# 1) 통과할 주파수 대역을 정의하는 마스크를 생성한다.
rows, cols = img.shape
crow, ccol = rows//2, cols//2 # 이미지의 중심 좌표. //은 나눗셈 몫을 반환하는 연산자.

d = 1   # 일단 중앙부 근처의 값을 실험삼아 출력해 본다. -> 크기가 어느 정도인지 확인해 보고자 함.
print(abs(fshift[crow-d:crow+d+1, ccol-d:ccol+d+1]))

# 마스크를 외부 모듈의 함수로 생성한다.
#mask_blr, mask = me.get_filter_mask((rows, cols), d=30, option='circle', sigma=5)
mask_blr, mask = me.get_filter_mask((rows, cols), d=30, option='square', sigma=5)

# 2) 생성된 마스크 2종을 화면에 출력한다.
plt.subplot(242)
plt.imshow(mask, cmap='gray')
plt.title('(b) mask', fontsize=15)   # blur처리 안된 마스크
plt.xticks([]), plt.yticks([])

# blurring 처리된 마스크. ringing effect 감소 효과가 있음
plt.subplot(246)
plt.imshow(mask_blr, cmap='gray')
plt.title('(f) mask_blr', fontsize=15)
plt.xticks([]), plt.yticks([])

# -------------------------------------------------------------------------------------------
# 단계 4: 필터링 실시
# 제작된 마스크와 푸리에 변환 결과를 곱하여 변형된 푸리에 결과를 만든다.
# 마스크를 통과하지 못한 주파수 영역은 0으로 된다.
# -------------------------------------------------------------------------------------------

# 1) 푸리에 변환 결과에 마스크를 곱하여 통과하거나 차단할 영역을 결정한다.
fshift_m = fshift * mask          # 마스크를 블러링하지 않은 것
fshift_mb = fshift * mask_blr      # 마스크를 블러링한 것

# 2) 마스크와 곱셈동작으로 처리된 스펙트럼 2종을 도시한다.
spectrum_m = 20 * np.log10(np.abs(fshift_m)+1)     # 가공된(필터링된) 영역에 대한 스펙트럼
plt.subplot(243)
plt.imshow(spectrum_m, cmap='gray')
plt.title('(c) spectrum_m=Spectrum(e)*mask(b)', fontsize=15)
plt.xticks([]), plt.yticks([])

spectrum_mb = 20 * np.log10(np.abs(fshift_mb)+1)     # 가공된(필터링된) 영역에 대한 스펙트럼
plt.subplot(247)
plt.imshow(spectrum_mb, cmap='gray')
plt.title('(g) spectrum_mb=Spectrum(e)*mask_blr(f)', fontsize=15)
plt.xticks([]), plt.yticks([])


# 3) 마스크 처리된 푸리에 변환 결과를 다시 자신의 위치로 재 배치
# ifftshift() : 대각 대칭이동된 데이터를 다시 본래 위치로 복귀 시킨다.
f_ishift_m = np.fft.ifftshift(fshift_m)
f_ishift_mb = np.fft.ifftshift(fshift_mb)

# 2) raw data 형식으로 재배치된 푸리에 결과의 스펙트럼을 보인다.
plt.subplot(244)
plt.imshow(np.fft.ifftshift(spectrum_m), cmap='gray')
plt.title('(d)ifftshift(spectrum_m)', fontsize=15)
plt.xticks([]), plt.yticks([])

plt.subplot(248)
plt.imshow(np.fft.ifftshift(spectrum_mb), cmap='gray')
plt.title('(h)ifftshift(spectrum_mb)', fontsize=15)
plt.xticks([]), plt.yticks([])



# -------------------------------------------------------------------------------------------
# 단계 5: 변경된 푸리에 변환결과를 다시 이미지로 변환
# -------------------------------------------------------------------------------------------
# 1) 영상으로 역변환한다. 그러나 복소형 데이터라서 출력할 수 없다.
plt.figure(num='input & output', figsize=(11, 8))
#plt.subplots_adjust(top=0.95, bottom=0.05)
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.2, right=0.8, hspace=0.1)  # hspace 그림간 height 간격
plt.gray()      # cmap='gray' 지정 안해도 됨.

img_m = np.fft.ifft2(f_ishift_m)
img_mb = np.fft.ifft2(f_ishift_mb)
print("img_m(inverse & filtered): ", type(img_m), img_m.dtype, img_m.itemsize)

plt.subplot(221)    # 필터링된 역변환 영상
img1 = np.abs(img_m)
#plt.imshow(img1, cmap='gray')
plt.imshow(img1)
plt.title('(a) abs(), filtered with mask', fontsize=15)
plt.axis('off')


plt.subplot(222)    # 필터링된 역변환 영상
img2 = np.real(img_m)
plt.imshow(img2)
plt.title('(b) real(), filtered with mask', fontsize=15)
plt.axis('off')


plt.subplot(223)
img3 = np.abs(img_mb)
plt.imshow(img3)
plt.title('(c) abs(), filtered with mask_blr', fontsize=15)
plt.axis('off')


plt.subplot(224)
img4 = np.real(img_mb)
plt.imshow(img4)
plt.title('(d) real(), filtered with mask_blr', fontsize=15)
plt.axis('off')

plt.show()

