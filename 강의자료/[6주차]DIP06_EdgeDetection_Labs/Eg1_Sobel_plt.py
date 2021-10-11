"""
****  다음 설명문은 Eg1_Sobel_cv.py + Eg1_Sobel_plt.py  두 예제 프로그램에 공통으로 적용된다. ***

1. 개요
    OpenCV 함수 Sobel를 이용하여 에지를 검출한 결과를 출력한다.
    편의상 윈도의 크기를 정하고 인터랙티브 모드로 동작하도록 설정하였다.

    !!!! pyplot.imshow()을 사용할 때의 중요 참고 사항 !!!!!
    1) mono gray 영상을 출력할 때는 cmap의 설정에 따라 영상 데이터의 표현 기법이 달라진다.
        cmap은 0~255의 gary 정보를 지정된 pseudo(의사) color로 표현한다.
        아무것도 지정하지 않으면 사용되는 default color map은 cmap='viridis'이다.
        mono gray로 표현하고자 할 때는 cmap='gary'로 지정한다.
    2) pyplot.imshow()는 영상 데이터의 최대, 최소를 구해 출력할 영상의 데이터 범위를 자동 스케일링한다.
        내부에서 최소, 최대 값을 구해 자동 스케일링을 행한다.
        그래서 에지 데이터에 대해 음수는 0~0,5, 양수는 0.5~1의 범위를 갖게된다.
        모노 영상에 대해서는 자동 스케일링을 행하지만, 칼라 영상에 대해서는 자동 스케일링을 행하지 않는다.

    # pyplot.imshow()의 장점
        cv.imshow()와는 달리 프로그래머는 수의 범위를 고려하지 않고 출력해도 된다.
        부호가 있는 수를 표현할 때 매우 편리하다.
        cmap 파라미터의 설정에 따라 매우 다양한 영상 출력을 시도할 수 있다.
            여기서는 cmap='gray'만 사용하였다. 개인적으로 따로 공부하기 바람.

2. 다음 내용을 숙지했는지 점검하기 바람
    1) cv.Sobel() 함수의 기본 사용법을 숙지한다.

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
#Name = 'circles_rects.png'

FullName = Path + Name

img = cv.imread(FullName, 0)                # 주의 !!! 모노 영상. 에지 영상은 그레이 영상에 대해 적용하는 것이 의미가 있음.
print("cv2: type(img)=", type(img), " img.dtype=", img.dtype, "img.shape=", img.shape)

# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert img is not None, f'No image file [{Name}]....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

img = img / 255

def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    # 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
    #plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
    #plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # 상하좌우 여백의 크기를 지정

#"""
# ----------------------------------------------------------------------------------------------------------------------
# 화면 1 - matplotlib.pyplot.imshow()를 이용한 처리 결과 보이기
# cv.Sobel() 함수를 이용해 소벨 필터링을 행한다.
# 화면을 2x2로 나누어 1) 원본, 2) xy 모두의 에지, 3) x축 방향의 에지, 4) y축 방향의 에지를 보인다.
# 이때 필터링 결과는 음수의 크기가 클수록 어둡게, 양수가 클 수록 밝게 표시된다.
# 이러한 처리는 pyplot.imshow() 함수의 내부 크기 및 부호 정규화 기능에 의존하였다.
# ----------------------------------------------------------------------------------------------------------------------

fig = plt.figure(num='1. original and scaled Sobel values', figsize=(13, 8))
#fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.3)

scale = 2
plt.suptitle(f"1. Original image and Sobel Values, scale={scale:#.1f}", fontsize=12, fontweight='bold')

#maximizeWindow()


# dst=cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]
k = 3     # 커널 크기
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)    # x축 에지
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)    # y축 에지

# https://matplotlib.org/tutorials/colors/colormaps.html
plt.subplot(2, 2, 1)
ttle1 = f"{np.min(img):#.1f} <= original img <= {np.max(img):#.1f}"
plt.title(ttle1)
#plt.title(f'Original, scale={scale:#.1f}')               # 원본 영상
plt.imshow(scale * img, cmap='gray')
#plt.imshow(scale * img.astype(np.float64), cmap='gray')
#plt.imshow(img)                         # default color map
#plt.imshow(img, cmap='viridis')         # default color map
#plt.imshow(img, cmap=None)              # default color map
#plt.axis('off')                     # 테두리 없음.
plt.xticks([]), plt.yticks([])      # 검은 테두리 있음

# 내부에서 정규화를 행하기 때문에 scale를 곱해도 결과는 같다.
plt.subplot(2, 2, 3), plt.imshow(scale*sobelx, cmap='gray')
ttle3 = f"{np.min(scale*sobelx):#.1f} <= scale * Sobel x <= {np.max(scale*sobelx):#.1f}"
plt.title(ttle3), plt.xticks([]), plt.yticks([])
print(ttle3)


plt.subplot(2, 2, 4),plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
print(f"{np.min(scale*sobely)} <= scale * Sobel y <= {np.max(scale*sobely)}")

# 절댓값을 취하지 않아 x방향, y방향 에지 성분이 서로 상쇄될 수 있다.
# 출력 결과를 보면 대각선 방향의 에지 성분이 약화된 모습을 볼 수 있다.
sobel_xy = scale * (sobelx + sobely)
plt.subplot(2, 2, 2)
plt.imshow(scale*sobel_xy, cmap='gray')
plt.title('Sobel in x and y directions')
plt.xticks([]), plt.yticks([])      # 검은 테두리 있음
print(f"{np.min(scale * (sobelx + sobely))} <= scale * (sobelx + sobely) <= {np.max(scale * (sobelx + sobely))}")


# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # 상하좌우 여백의 크기를 지정

plt.waitforbuttonpress()

# ----------------------------------------------------------------------------------------------------------------------
# 화면 2 - matplotlib.pyplot.imshow()를 이용한 처리 결과 보이기
# cv.Sobel() 함수를 이용해 소벨 필터링을 행한다.
# 화면을 2x2로 나누어 1) 원본, 2) xy 모두의 에지, 3) x축 방향의 에지, 4) y축 방향의 에지를 보인다.
# 실험 1과 다른 점
#   이때 필터링 결과는 부호를 고려하지 않고 np.abs() 함수를 사용하여 모두 양수로 변환하여 사용한다.
# ----------------------------------------------------------------------------------------------------------------------

plt.figure(num='2. Sobel Edge with sign neglected', figsize=(13, 8))
#maximizeWindow()
plt.suptitle("2. abs(Sobel)", fontsize=12, fontweight='bold')

# dst=cv.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]
sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)    # x축 에지
sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)    # y축 에지

scale = 3
# https://matplotlib.org/tutorials/colors/colormaps.html
plt.subplot(2, 2, 1)
plt.title('Original')               # 원본 영상
plt.imshow(img, cmap='gray')  # 임의의 수를 곱해도 결과는 같다. 이유를 설명하시오.
plt.xticks([]), plt.yticks([])      # 검은 테두리 있음

# 임의의 수를 곱해도 결과는 같다. 이유를 설명하시오.
plt.subplot(2, 2, 3), plt.imshow(np.abs(scale*sobelx), cmap='gray')
plt.title('|Sobel X|'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 2, 4),plt.imshow(np.abs(scale*sobely), cmap='gray')
plt.title('|Sobel Y|'), plt.xticks([]), plt.yticks([])

sobel_xy = np.abs(sobelx) + np.abs(sobely)
#sobel_xy = sobelx + sobely
plt.subplot(2, 2, 2), plt.imshow(scale*sobel_xy, cmap='gray')
plt.title('Sobel in |x| and |y| directions')
plt.xticks([]), plt.yticks([])      # 검은 테두리 있음
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
#plt.subplots_adjust(top=0.9, bottom=0.1) # 맨 나중에 쓴 것에 따라 위의 설정이 바뀔 수 있음.
#plt.show()
plt.waitforbuttonpress()



# ----------------------------------------------------------------------------------------------------------------------
# 참고: plt exercise- color과 mono 스케일링 다시 확인 하기
# 영상 데이터의 값에 전체적 변화가 일어나면?
#   칼라에 대해서는 정규화 일어나지 않음.
#   모노에 대해서는 정규화가 일어남. 밝아지지 않음.
# ----------------------------------------------------------------------------------------------------------------------

plt.figure(num='3. automatic normalization for color?', figsize=(13, 8))
#maximizeWindow()
plt.suptitle("3. automatic normalization for color?", fontsize=12, fontweight='bold')

img = cv.imread(FullName)
b, g, r = cv.split(img)
imgC = cv.merge([r, g, b]) / 255


plt.subplot(131)
plt.imshow(imgC)
plt.title('Original')
plt.xticks([]), plt.yticks([])    # 검은 테두리 있음

scale = 3.5
plt.subplot(132)
plt.imshow(scale * imgC)        # 칼라에 대해서는 정규화 일어나지 않음.
plt.title('Brightened Color: scale=' + str(scale))
plt.xticks([]), plt.yticks([])    # 검은 테두리 있음

plt.subplot(133)
#imgM = cv.cvtColor(img, cv.COLOR_BGR2GRAY)         # 이상없음
#imgM = cv.cvtColor(imgC, cv.COLOR_RGB2GRAY)        # 오류: 'depth' is 6 (CV_64F)
imgM = cv.cvtColor(imgC.astype(np.float32), cv.COLOR_RGB2GRAY)
plt.imshow(scale*imgM, cmap='gray')     # 모노에 대해서는 정규화가 일어남. 밝아지지 않음.
plt.title('Brightened Mono: scale=' + str(scale))
plt.xticks([]), plt.yticks([])      # 검은 테두리 있음
plt.tight_layout() # 맨 나중에 써야 영향을 미침.
#plt.subplots_adjust(top=0.9, bottom=0.1) # 맨 나중에 쓴 것에 따라 위의 설정이 바뀔 수 있음.

plt.waitforbuttonpress()
#plt.show()     interactive mode에서는 사용하면 안됨.



