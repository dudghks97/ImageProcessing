"""

1. 프로그램의 기능
    Unsharp Masking 기법으로 영상의 선명도를 강화하는 알고리즘을 소개한다.

2. 프로그램의 목표
    입력 영상에 대해;
    1) 블러된 영상을 구하고,
    2) 이를 입력 영상과 차감하여 고주파 영상을 구하여,
    3) 원본 영상에 고주파 영상을 더하여 영상의 선명도를 강화한 영상을 획득한다.

3. 질문
    1) unsharp masking에 의한 영상 화질 강화의 원리는 무엇인지 설명하시오.
    2) 처리 단계에서 먼저 blur 영상을 구하는데 이때 블러링을 통제하는 시그마의 역할에 대해서 설명하시오.
    (즉, 시그마의 크기가 화질에 미치는 영향은?)

4. 미션
     시그마와 고주파 강화용 스케일을 조정하는 2개의 트랙바를 설치하여 이들을 조정하는 Unsharp Masking 프로그램을 설계하시오.

"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def maximizeWindow():     # Win10에서 창크기 최대화하는 방법
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

#------------------------------------------------------------------------------------
# 단계 0 : 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Name = 'bone(m).bmp'
Name = 'lenna.tif'
#Name = 'monarch.png'
#Name = 'fruits.jpg'
#Name = 'Fig0303(a)(breast).tif'
#Name = 'tank.bmp'
#Name = 'HazyTown.jpg'

FullName = Path + Name
img = cv.imread(FullName)

assert img is not None, "Failed to load image file.."

img = img/255


# =================================================================================================================
# 실습 1: Unsharp Masking의 처리과정을 단계적인 수행 결과와 함께 보인다.
# =================================================================================================================

#------------------------------------------------------------------------------------
# 단계 1 : 블러링 영상(blur)을 구한다.
#------------------------------------------------------------------------------------

blur=cv.GaussianBlur(src=img, ksize=(21,21),
        sigmaX=3, borderType=cv.BORDER_REPLICATE )


#------------------------------------------------------------------------------------
# 단계 2 : 영상의 고주파 성분(UnsharpMaskImg)을 구하여 화면에 출력한다.
#------------------------------------------------------------------------------------
UnsharpMaskImg = img - blur # 고주파 영상

# -를 포함하는 성분을 임의 숫자 0.5를 더해서 +로 환원해서 출력한다.
cv.imshow('High Frequency Image + 0.5', UnsharpMaskImg + 0.5 )

# pyplot는 0을 중간 값을 만들어 -를 포함하는 영상을 적절히 표현한다.
print("Close the plt window to continue...")
plt.figure(num='3*UnsharpMaskImg'); plt.imshow(3*UnsharpMaskImg)
plt.axis('off')         # plt.xticks([]), plt.yticks([]);


#------------------------------------------------------------------------------------
# 단계 3 : 입력 영상에 고주파 성분을 더하여 대조비 강화 영상을 만든다.
#------------------------------------------------------------------------------------
#"고주파 성분 = 원본 영상 - 블러링 된 영상"이라 할 수 있다.
# 고주파성분을 강조하기 이해 적당히 스케일링하였다.
scale = 1; SharpenedImg = img + scale * UnsharpMaskImg

# 영상 어레이를 수평으로 연결하는 함수: np.hstack, cv.hconcat - 이 두개의 함수의 동작은 동일하다.
shw = np.hstack((img, SharpenedImg))            # stacking images side-by-side
#shw = cv.hconcat((img, SharpenedImg))
cv.imshow('Original Image & Sharpened Image', shw)


plt.show()
cv.waitKey(0)
#exit(0)



# =================================================================================================================
# 실습 2: 시그마와 강도 조절 효과를 관찰한다.
# 1) 시그마의 역할을 확인할 수 있도록 여러 결과를 한 화면에 출력한다.
# 2) 지정된 시그마에 대한 강도를 다양하게 적용한다.
# =================================================================================================================

b, g, r = cv.split(img); img = cv.merge([r, g, b]);

plt.figure(num='Unsharp Masking: Various Sigma')
maximizeWindow()
plt.suptitle("Fixed sacle & various sigma", fontsize=15, fontweight='bold')

plt.subplot(231)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

scale = 2
for i, sigma in enumerate([1, 3, 5, 7, 13]):
        plt.subplot(232+i)
        k = sigma * 6 + 1
        um = img + scale * (img - cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=sigma))
        #plt.imshow(um)
        # Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
        plt.imshow(np.clip(um, 0, 1))
        plt.title(f'sigma={sigma}, strength={scale}')
        plt.axis('off')




plt.figure(num='Unsharp Masking: Various Strength')
maximizeWindow()
plt.suptitle("Fixed sigma & various scale", fontsize=15, fontweight='bold')

plt.subplot(231)
plt.imshow(img)
plt.title('Original')
plt.axis('off')

sigma = 13
for i, scale in enumerate([1, 2, 3, 4, 5]):
        plt.subplot(232+i)
        k = sigma * 6 + 1
        um = img + scale * (img - cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=sigma))
        #plt.imshow(um)
        plt.imshow(np.clip(um, 0, 1))
        plt.title(f'sigma={sigma}, strength={scale}')
        plt.axis('off')





plt.show()
exit(0)