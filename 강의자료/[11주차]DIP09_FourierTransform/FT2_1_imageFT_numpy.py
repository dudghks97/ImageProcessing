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
# 실험 1 : Numpy를 이용한 푸리에 변환
# Fourier Transform을 적용.
# 적용을 하면 0,0, 즉 화면 좌측상단점이 중심이고, 거기에 저주파가 모여 있음.
# 분석을 용이하게 하기 위해 0,0을 이미지의 중심으로 이동 시키고 Log Scaling을 하여 분석이 용이한 결과값으로 변환
#======================================================================================================

# 단계 1: 원본 영상을 보인다.
plt.figure(num='Fourier Transform Test', figsize=(15, 9))
plt.subplot(231)        # 원본 영상
plt.imshow(img, cmap='gray')
plt.title('Original')
plt.xticks([]), plt.yticks([])

# 단계 2: 변환된 데이터를 역변환하여 같은 결과가 생성되는지 관찰한다.
plt.subplot(232)        # 역변환 영상
f = np.fft.fft2(img) # 이미지에 푸리에 변환 적용
inv_f = abs(np.fft.ifft2(f))
print(f'Input: dtype of={img.dtype}, shape={img.shape}')
print(f'Transform: dtype of={f.dtype}, shape={f.shape}')
print(f'Inverse transform: dtype of={inv_f.dtype}, shape={inv_f.shape}')

plt.imshow(inv_f, cmap='gray')
plt.title('inverse transformed')
plt.xticks([]), plt.yticks([])

# 역변환결과와 원본이 얼마나 같은지 비교한다.
# PSNR은 두 영상의 같음을 픽셀레벨로 비교하는 지표이다.
# 40[dB]이상이면 다르다고 느끼지 쉽지 않다.
print(img.dtype)    #uint8
psnr = cv.PSNR(img, np.clip(inv_f, 0, 255).astype('uint8'))
print(f'PSNR={psnr:#.1f}[dB]')    # PSNR= 56.20630266375299


# 단계 3: 푸리에 변환 결과(파워 스펙트럼)를 위치를 조정하지 않고 그대로 보인다.
# 주파수가 0인 부분이 꼭지점에 위치한다.
plt.subplot(233)        # 파워스펙트럼 영상
# spectrum 구하는 수학식. +1은 log(0)를 방지하기 위한 조처.
spectrum = 20 * np.log10(np.abs(f)+1)
plt.imshow(spectrum, cmap='gray')
plt.title('spectrum(raw)'), plt.axis('off')

# 단계 4: 중앙에 저주파가 모인 파워 스펙트럼을 보인다.
# 분석을 용이하게 하기 위해 주파수가 0인 부분을 중앙에 위치시킴.
plt.subplot(234)        # shifted 파워스펙트럼 영상
fshift = np.fft.fftshift(f)
spectrum2 = 20 * np.log10(np.abs(fshift)+1)
plt.imshow(spectrum2, cmap='gray')
plt.title('spectrum(shifted)'), plt.axis('off')

# 단계 5: 중앙에 저주파가 모인 false colored 파워 스펙트럼을 보인다.
plt.subplot(235)        # 파워스펙트럼 영상
spectrum_s = np.fft.fftshift(spectrum)
plt.imshow(spectrum_s)
plt.title('spectrum(shifted) using color map'), plt.axis('off')

# 단계 6: Phase angle(위상각)을 보인다.
plt.subplot(236)        # 위상각 영상
angle = np.angle(f)
plt.imshow(angle, cmap='gray')
plt.title('phase angle'), plt.axis('off')


plt.show()
