"""
1. 동작 개요
    1) resize(): 영상의 크기를 변화시킨다.
    2) flip(): 영상을 상하좌우로 뒤집는다.
    3) cv.hconcat(), cv.vconcat(): 크기가 같은 두개 영상을 좌후, 상하로 이어 붙인다.


2. 주요 함수
    1) 영상의 크기를 변경하는 함수, 조절된 영상 = resize()
        확대, 축소하는 알고리즘을 지정할 수 있다. interpolation=cv.INTER_LINEAR
        고정된 크기 지정 방식: cv.resize(영상, (가로, 세로))
        비율(fx, fy)로 지정하기: cv.resize(img, dsize=(0, 0), fx, fy)
    2) 영상 flipping: flip
        cv.flip(img2, 0)     # vertical flip
        cv.flip(img2, 1)     # horizontal flip
        cv.flip(img2, -1)    # vertical & horizontal flip
    3) 영상 이어붙이기:
        c = cv.hconcat([a, b])      # 같은 크기의 영상 어레이 a, b를 가로로 이어 붙여 반환한다.
        c = cv.vconcat([a, b])      # 같은 크기의 영상 어레이 a, b를 세로로 이어 붙여 반환한다.
        c = numpy.concatenate(...)

3. 질문

"""

# ========================================================================================================
# 섹션 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ========================================================================================================
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'd:/StudyImages/Images/'        # 사용가능
Path = 'd:/Work/StudyImages/Images/'
Path = '../data/'

Name = 'lenna_mono.jpg'     # mono gray 영상. 1채널 영상
Name = 'RGBColors.JPG'
Name = 'lenna.tif'
Name = 'chelsea.png'
#Name = 'monarch.bmp'


FullName = Path + Name


import cv2 as cv
import numpy as np

# ========================================================================================================
# 공통 : 영상 파일을 읽어 들이기
# ========================================================================================================
img = cv.imread(FullName, -1)       # read as is
assert img is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
print(f"shape of original image={img.shape}")
cv.imshow(f'0) original:{img.shape}', img)

#"""
# ========================================================================================================
# 실습 1: 영상의 크기를 변경하는 함수, resize()
# 예: cv.resize(영상, (가로*세로)) 영상을 (가로*세로) 크기로 조절하고 싶을 때.
# 주의 사항:
# resize()에 사용하는 사각형의 정보는 width x height이다. row x col을 사용하는 ndarray.shape와 반대이다.
# ========================================================================================================

# 1) 타깃의 크기를 지정하는 방식: 고정된 크기로 조정
width, height = 640, 280    # 크기를 지정하여 조정
resized_img = cv.resize(img, (width, height))
cv.imshow(f'1) fixed: {img.shape}->{resized_img.shape}', resized_img)

# 2) 타깃의 크기를 지정하는 방식: 자체의 속성 정보를 바탕으로 조정. 가로는 늘리고(x1.5) 세로는 줄여(//2) 본다.
width, height = int(img.shape[1]*1.5), (img.shape[0])//2    # //은 몫을 구하는 연산자. int()함수를 써도 같은 결과.
resized_img = cv.resize(img, (width, height))
cv.imshow(f'2) wxh=1.5x0.5: {img.shape}->{resized_img.shape}', resized_img)

# 3) 가로, 세로 비율로 조정하는 방식. interpolation 알고리즘도 선택해 본다.
fx_r = 2; fy_r = 2
resized_img = cv.resize(img, dsize=(0, 0), fx=fx_r, fy=fy_r, interpolation=cv.INTER_LINEAR)
cv.imshow(f'3) propotional: fx={fx_r:#.1f}, fy={fy_r:#.1f}, {img.shape}->{resized_img.shape}', resized_img)

# 4) 다른 영상(img)와 같은 크기로 조정(잘못된 방법).
# 그런데 이런 방식으로 하면 세로, 가로 정보가 뒤바뀐다. --- 주의!!! 가로 x 세로
img2 = cv.imread(Path + "colorbar_chart.jpg", -1)       # read as is
print(f'target shape={img2.shape}')
resized_img = cv.resize(img2, img.shape[0:2])
cv.imshow(f'4) Incorrect: {img2.shape}->{resized_img.shape}', resized_img)

# 5) 다른 영상(img)와 같은 크기로 조정(올바른 방법).
# ndarray.shape를 사용하려면 row x col의 순서를 바꾸어 주어야 한다.
resized_img = cv.resize(img2, img.shape[1::-1]) # shape[2], shape[1]이 순서대로 만들어진다.=> 가로x세로
cv.imshow(f'5) As targeted: {img2.shape}->{resized_img.shape}', resized_img)

cv.waitKey(0)
cv.destroyAllWindows()
#exit(0)
#"""




# ========================================================================================================
# 실습 2: 영상 flipping & image stacking
# ========================================================================================================

# 1) image flipping
img_flip_along_x = cv.flip(img, flipCode=0)     # vertical flip. y(세로) 방향의 flip
img_flip_along_y = cv.flip(img, flipCode=1)     # horizontal flip. x(가로) 방향의 flip
img_flipped_xy = cv.flip(img, flipCode=-1)      # vertical & horizontal flip

cv.imshow('1) original image', img)
cv.imshow('2) img_flip_along_x: vertical flip', img_flip_along_x)
cv.imshow('3) img_flip_along_y: horizontal flip', img_flip_along_y)
cv.imshow('4) img_flipped_xy', img_flipped_xy)
cv.waitKey(0)
cv.destroyAllWindows()


# 2) image stacking, 영상 이어붙이기
# To stack horizontally (img1 to the left of img):
#   vis = np.concatenate((img1, img), axis=1)  # axis=1. x축(가로 방향)상으로 이어붙이기
# To stack vertically (img1 over img):
#   vis = np.concatenate((img1, img), axis=0)  # axis=0. y축(세로 방향)상으로 이어붙이기

# 각각 2개씩 가로 방향으로 이어 붙인다.
upper = np.concatenate((img, img_flip_along_y), axis=1)        # axis=1. x축(가로 방향)상으로 이어붙이기
lower = np.concatenate((img_flip_along_x, img_flipped_xy), axis=1)

# 위 2개의 영상을 세로로 이어 붙인다.
vis = np.concatenate((upper, lower), axis=0)

# 영상이 너무 커서 4개의 영상이 묶인 vis 영상을 가로, 세로 각각 1/2 축소하여 출력한다.
width, height = int(vis.shape[1]/2), int(vis.shape[0]/2)
vis2 = cv.resize(vis, (width, height))

cv.imshow('vis2', vis2)
cv.waitKey(0)
cv.destroyAllWindows()

two_h = cv.hconcat([img, img_flip_along_x])     # 2 어레이의 데이터 형이 같아야 한다.
two_v = cv.vconcat([img, img_flip_along_x])
cv.imshow('two_h', two_h)
cv.imshow('two_v', two_v)

cv.waitKey(0)

