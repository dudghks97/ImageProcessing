"""
참고 URL
        https://www.imageeprocessing.com/2011/10/add-salt-and-pepper-noise-to-image.html
1. 프로그램의 기능
    메디언 필터링으로 salt & pepper 잡음을 제거한다.

2. 프로그램의 목표
    메디언 필터링이 소금 후추 잡음 제거에 강한 이유를 이해한다.
    커널의 크기가 미치는 영향을 이해한다.
    메디언 필터는 칼라 영상에 대해서도 적용 가능하다.

3. 질문
    메디언 필터가 S & P 잡음처리에 우수한 이유를 설명하시오.
    메디언 필터는 가우시안 잡음 처리에 대해서는 효과적이지 못하다. 그 이유를 설명하시오.


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
Name = 'bone(m).bmp'
Name = 'lenna.tif'
#Name = 'monarch.png'
#Name = 'fruits.jpg'
Name = 'BAR.bmp'



FullName = Path + Name
imgOrg = cv.imread(FullName, 0)
assert imgOrg is not None, "Failed to load image file.."

cv.imshow('step 0: original image', imgOrg)


#------------------------------------------------------------------------------------
# 단계 1 : 아래 프로그램으로 만들어진 잡음이 첨가된 영상을 읽어 들인다.
# 잡음 영상 만드는 프로그램: Eh7_.AddNoise_imnoise.py를 참고 바람
# 잡음 영상을 현재의 폴더에 만들기 때문에 미리 위 프로그램을 수행하여 현재 폴더에 잡음 영상을 만들어야 한다.
#------------------------------------------------------------------------------------

# 현재의 폴더에서 소금 후추 잡음 영상을 읽어들인다.
img1 = cv.imread(Name+'_sp.png')
assert img1 is not None, "Salt & pepper image is not loaded..."
cv.imshow('step 1: noisy image(salt & pepper)', img1)

# 가우스 잡음 영상을 읽어들인다.
img2 = cv.imread(Name+'_gauss.png')
assert img2 is not None, "Salt & pepper image is not loaded..."
cv.imshow('step 1: noisy image(gauss)', img2)

#=================================================================
# 단계 2 : medianBlur() 함수를 이용하여 메디언 필터링을 행한다.
#=================================================================

md = cv.medianBlur(src=img1, ksize=3)
cv.imshow('step 2: S&P- median filtered, k=3', md)

md = cv.medianBlur(src=img1, ksize=5)
cv.imshow('step 2: S&P- median filtered, k=5', md)

md = cv.medianBlur(src=img2, ksize=3)
cv.imshow('step 2: GAUSS- median filtered, k=3', md)

md = cv.medianBlur(src=img2, ksize=5)
cv.imshow('step 2: GAUSS- median filtered, k=5', md)

cv.waitKey()
exit(0)
