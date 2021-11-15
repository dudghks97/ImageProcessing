"""
1. 프로그램의 기능
    1D FFT를 행하고 그 결과를 도시한다.

2. 프로그램의 목표
    1차원 신호의 FFT 변환 값의 특성을 이해한다.

3. 프로그램에서 점검 포인트
    numpy 함수를 사용한 푸리에 변환 함수의 사용법을 점검한다.

4. 미션
    사례 함수에 대한 주파수 변환 결과를 도시하시오.

5. 질문


참조 URL

1D - FFT(python)
    https://techreviewtips.blogspot.com/2017/11/05-02-fft.html
1D - FFT(matlab)
    https://kr.mathworks.com/help/matlab/ref/fft.html
    Matlab source: o_09_02_FourierTransform1D.m

Bandwidth fn (= Nyquist frequency).
    이 값은 주어진 샘플링 주파수에 의해 FFT에 의해 결정될 수 있는 이론상 최대 주파수를 나타냅니다.
    fn = fs / 2
    예를 들어 48kHz의 샘플링 속도에서 이론적으로 24kHz까지의 주파수 성분을 결정할 수 있습니다.
    https://www.nti-audio.com/ko/%EB%89%B4-%EC%8A%A4/fft%EC%97%90-%EA%B4%80%ED%95%9C-%EB%AA%87-%EA%B0%80%EC%A7%80-%EC%82%AC%ED%95%AD%EC%9D%84-%EC%A0%95%EB%A6%AC%ED%95%B4-%EB%B4%85%EC%8B%9C%EB%8B%A4

"""


import matplotlib.pyplot as plt
import numpy as np
import math


#============================================================================================
# 실험 1 : 1차원 신호에 대해 FFT를 취한 결과의 스펙트럼을 보인다.
# 양방향 스펙트럼을 보였다.
# Python 기반의 소스를 활용함. 위상차와 비교하여 볼 수 있음.
# 참조 : https://techreviewtips.blogspot.com/2017/11/05-02-fft.html
#============================================================================================

# 단계 1: 신호를 분석하기 위한 sampling speed와 관찰 시간의 구간을 정의한다.
Fs = 2000                   # Sampling frequency
T = 1/Fs                    # Sample interval time
te = 1                      # End of time
t = np.arange(0, te, T)     # Time vector
n = Fs * te                 # Number of samples
print(f'1. Sampling frequency={Fs}[Hz], Total time interval={te}[sec],',
      f'Number of samples={n}, Number of time vector={len(t)}')


# 단계 2: 분석할 입력 신호를 정의한다.

# 실험 1
A = 1           # amplitude
f = 50          # frequency
phi = 0         # phase shift
y = A * np.cos(2 * np.pi * f * t + phi)

# 실험 2
#y = A*np.cos(2*np.pi*f*t+phi)

# 실험 3
#y = np.sin(2*np.pi*f*t) + 1

# 실험 4
#f1 = 50; f2 = 100; y = 2 * np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# 실험 5
#f1=50; f2=100; y = 2 * np.sin(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t);

print(f'2. y: Amplitude={A}, Frequency={f}[Hz], Phase shift={phi}')


# 단계 3: 입력 신호 전체와 일부분을 그림으로 표현한다.
plt.figure(figsize=(15, 8), num='input signal and its Fourier Transform(magnitude & phase)')
plt.suptitle('input signal and its Fourier Transform(magnitude & phase)', size=15)
#plt.subplots_adjust(hspace=0.6, top=0.9, bottom=0.1, wspace=0.4)
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, hspace=0.6, wspace=0.3)
plt.subplot(321)        # 전체
plt.plot(t, y, 'r')
plt.title('input signal(full)', fontsize=15)
plt.xlabel('time[sec.]')        # plt.xlabel('time($sec$)')
plt.ylabel('y')
plt.xlim(0, te)
plt.grid()

plt.subplot(322)        # 일부분
plt.plot(t, y, 'r')
plt.xlim(0, 5/f)             # 5개의 사이클만을 확대하여 보인다. =개수*주기
plt.xlabel('time[sec.]')
plt.grid()
plt.ylabel('y')
plt.title(f'input signal(part)', fontsize=15)

# 단계 4: 푸리에 변환을 시행한다.
n = len(y)        # Length of signal
Y = np.fft.fft(y)/n     # fft computing and normalization. 데이터 샘플 수로 나누어 정규화한다.
print(f'4.1 len(Y)={len(Y)}, Y.dtype={Y.dtype}, Y.shape={Y.shape}')
spectrum = 2 * abs(Y)         # 정규화된 푸리에 변환에 대한 정규화된 Fourier Spectrum. x2가 필요.
print(f"4.2 len(spectrum)={len(spectrum)},"\
      , f"spectrum.dtype={spectrum.dtype}, spectrum.shape={spectrum.shape}")
#phase_ang = np.angle(Y)*180/np.pi

# 단계 5: 변환결과의 크기(스펙트럼)를 있는 그대로 도시한다. 직류 성분이 양단 끝에 존재한다.
plt.subplot(323)
plt.title('Fourier Spectrum, Y: Raw Output', fontsize=15)
plt.plot(spectrum, 'r')
plt.xlim(0, n)
plt.ylim(0, np.amax(spectrum)*1.2)
plt.xlabel('frequency[Hz]')
plt.ylabel('magnitude')
plt.grid()

# 단계 6: 주파수 0인 직류성분이 가운데 오도록 스펙트럼을 재배치하고 실제의 주파수 값을 적는다.
# np.fft.fftshift()
#     Shift the zero-frequency component to the center of the spectrum.
#     This function swaps half-spaces for all axes listed (defaults to all).
#     Note that y[0] is the Nyquist component only if len(x) is even

plt.subplot(324)
plt.title('Fourier Spectrum, Y: Frequency 0 centered', fontsize=15)
spectrum = np.fft.fftshift(spectrum)
f_mp = np.arange(-n//2, n//2, 1)    # - frequency ~ + frequency
plt.plot(f_mp, spectrum, 'r')
plt.xlim(-n//2, n//2)
plt.ylim(0, np.amax(spectrum)*1.2)
plt.xlabel('frequency[Hz]')
plt.ylabel('magnitude')
plt.grid()


# 단계 7: 스펙트럼의 실수부를 도시한다.
YY = np.fft.fftshift(Y)
reY = np.real(YY)
imY = np.imag(YY)

plt.subplot(325)
plt.title('Fourier Spectrum, Y: Real Part', fontsize=15)
plt.plot(f_mp, reY, 'r')
plt.xlim(-n//2, n//2)
plt.ylim(np.amin(reY)*1.2, np.amax(reY)*1.2)
plt.xlabel('frequency[Hz]')
plt.ylabel('magnitude')
plt.grid()

# 단계 8: 스펙트럼의 허수부를 도시한다.
plt.subplot(326)
plt.title('Fourier Spectrum, Y: Imaginary Part', fontsize=15)
plt.plot(f_mp, imY, 'r')
plt.xlim(-n//2, n//2)
plt.ylim(np.amin(imY)*1.2, np.amax(imY)*1.2)
plt.xlabel('frequency[Hz]')
plt.ylabel('magnitude')
plt.grid()

#plt.savefig("./test_figure2.png",dpi=300)
plt.show()
exit()


