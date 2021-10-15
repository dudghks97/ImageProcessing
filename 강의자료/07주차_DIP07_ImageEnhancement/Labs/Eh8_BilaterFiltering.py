"""

1. 프로그램의 기능
   양면형(bilateral) 필터는 영상의 윤곽을 보존하면서 세세한 잡음에 대해서만 가우시안 블러링을 수행하는 필터이다.
   이 동작을 표준 영상에 가우시안 잡음을 첨가한 영상에 대해 대해 적용하여 그 효과를 관찰해 보기로 한다.
   OpenCV 함수로 양면형 필터링을 행한다.
   이 프로그램은 사전에 Eh7_.AddNoise_imnoise.py에서 생성한 파일(확장자 *_gauss.png)이 현재의 수행 폴더에 있어야 한다.

2. 프로그램의 목표
    양면형(bilateral) 필터가 어떤 원리로 에지를 보존하면서 미세한 잡음을 제거하는 기능을 수행하는지 이해한다.

3. 함수 설명
    dst	= cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])

    ▪ sigmaColor
        - sigma for Range Filter. 주의: 가우시안 블러링을 위한 시그마 아니라 블러링 대상의 계조치의 차이 값을 말함.
        - 이 값보다 작은 화소들에 대해 그 차이만큼의 가중치로 sigmaSpace에 지정한 만큼의 블러링을 행단다.
        - 이 값(사실 중점 화소와 주변화소의 계조차이)이 0이면 해당 위치의 레인지 필터의 계수는 1이 된다.
          즉, 블러링 하지 않고 원본을 그대로 통과시킨다.
        - 이 값이 지나치게 크면 거의 모든 픽셀에 대해 sigmaSpace에 지정한 만큼의 블러링을 행단다.
          즉, 일반 가우시안 필터와 같은 블러링을 행한다.(경계 보존 효과의 특성이 살아나지 않는다.)
        - 결론적으로 중심 pixel과의 주변 화소의 계조 값의 차이가 이 임계치보다 작은
          값을 갖는 필터 내의 주변 화소는 블러링된다. => 이것보다 크지 않으면 블러링하지 않는다.
    ▪ sigmaSpace
        - sigma for Domain Filter.
        - 종래 가우시안 블러링과 같은 개념의 시그마. 클수록 영상의 블러링이 심해지며,
          이론적으로는 이 값이 커지면 필터의 크기(d)도 함께 커져야 한다.
        - 이 도메인 필터는 가우시안 블러링 필터라고 말할 수 있고 양면형 필터는
          도메인 필터 앞에 각 계수의 게인을 조절하는 레인지 필터가 달려 있는 것이라 이해할 수 있다.
    ▪ d
        - Diameter of each pixel neighborhood that is used during filtering.
        - 음수이면 이 값은 sigmaSpace로부터 자동 계산된다.
        => 크기를 지정하는 것과 자동 계산의 차이를 실험하려면 sigmaColor를 극단적으로 크게 하여
           일반 가우시안 블러링처럼 만들고 그 차이를 관찰하는 것이 좋다.


4. 질문
    1) 아래의 파라미터가 결과 영상에 미치는 요소를 설명하시오.
    - d, sigmaColor, sigmaSpace
    2) 가우시안 필터와 양면형 필터의 차이점은 무엇인가?

5. 미션
    위 파라미터를 조정할 수 있는 트랙바로 만들어 그 처리한 결과를 관찰하여 파라미터의 역할을 기술하시오.


"""
#import cv2 as cv
#import matplotlib.pyplot as plt
import numpy as np, cv2 as cv, matplotlib.pyplot as plt


#------------------------------------------------------------------------------------
# 단계 0 : 영상이 존재하는 폴더와 파일 이름을 지정하여 영상 파일 읽어 내기
# 다음 사례를 활용하여 영상 파일이 있는 폴더를 적절히 지정하시오.
#------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = '../data/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
#Name = 'bone(m).bmp'
#Name = 'bilateral.jpg'
Name = 'lenna.tif'
#Name = 'monarch.png'
#Name = 'fruits.jpg'
#Name = 'BAR.bmp'


FullName = Path + Name
imgOrg = cv.imread(FullName)

assert imgOrg is not None, "Failed to load image file.."
cv.imshow('original image', imgOrg)


#------------------------------------------------------------------------------------
# 단계 1 : 아래 프로그램으로 만들어진 잡음이 첨가된 영상을 읽어 들인다.
# 잡음 영상 만드는 프로그램: Eh7_.AddNoise_imnoise.py를 참고 바람
#------------------------------------------------------------------------------------

# 소금 후추 잡음 영상을 읽어들인다. => 여기에 양선형 필터를 적용하기는 부적절하다.
#img1 = cv.imread(Name+'_sp.png')
#assert img1 is not None, "Salt & pepper image is not loaded..."
#cv.imshow('noisy image(salt & pepper)', img1)

# 가우스 잡음 영상을 읽어들인다.
img2 = cv.imread(Name+'_gauss.png')
#img2 = imgOrg.copy()
assert img2 is not None, "Salt & pepper image is not loaded..."
cv.imshow('noisy image(gauss)', img2)

#=================================================================
# 단계 2 : bilateralFilter() 함수를 이용하여 필터링을 행한다.
# 필터의 크기/블러할 픽셀값의 차이 한도/공간필터의 시그마
#=================================================================
#dst	= cv.bilateralFilter(img2, d=-1, sigmaColor, sigmaSpace, borderType=cv.BORDER_REPLICATE)
#d=-1; sigmaColor =50; sigmaSpace = 0.5
d=-1; sigmaColor = 1; sigmaSpace = 7
dst	= cv.bilateralFilter(img2, d, sigmaColor, sigmaSpace)
cv.imshow('BF1: d='+str(d)+', Color='+str(sigmaColor)+', Space='+str(sigmaSpace), dst)

d=-1; sigmaColor = 50; sigmaSpace = 7
dst	= cv.bilateralFilter(img2, d, sigmaColor, sigmaSpace)
cv.imshow('BF2: d='+str(d)+', Color='+str(sigmaColor)+', Space='+str(sigmaSpace), dst)

d=-1; sigmaColor = 100; sigmaSpace = 7
dst	= cv.bilateralFilter(img2, d, sigmaColor, sigmaSpace)
cv.imshow('BF3: d='+str(d)+', Color='+str(sigmaColor)+', Space='+str(sigmaSpace), dst)

#d=-1; sigmaColor = 77771900; sigmaSpace = 7
sigmaColor = 77771900; sigmaSpace = 7; d = sigmaSpace * 6 + 1
dst	= cv.bilateralFilter(img2, d, sigmaColor, sigmaSpace)
cv.imshow('BF4: d='+str(d)+', Color='+str(sigmaColor)+', Space='+str(sigmaSpace), dst)

#blur = cv.GaussianBlur(img2, (0, 0), sigmaSpace)       # 정해주는 커널 크기로 연산
blur = cv.GaussianBlur(img2, (d, d), sigmaSpace)  # 자신이 커널 크기 지정
psnr = cv.PSNR(dst, blur); print('psnr=', psnr);  # psnr= 57.55934569838967
cv.imshow('GaussianBlur sigmaSpace=' + str(sigmaSpace), blur)

cv.waitKey()
exit(0)
