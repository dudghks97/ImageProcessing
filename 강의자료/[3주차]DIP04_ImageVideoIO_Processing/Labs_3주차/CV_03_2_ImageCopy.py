"""
1. 개요
    영상 어레이를 복사한 후 원본을 수정하였을 때 복사본에 어떤 현상이 일어나는지 관찰한다.
    => =(assign 문장)에 의해 생성한 어레이의 경우 복사본/원본의 수정이 원본/복사본의 훼손 문제가 발생하는 경우가 있다.
    => 지금까지 관찰된 바로는 어레이 인덱스[ ]를 사용한 수정 작업에서 발견되었다. => 단계 3 실험 참조.
    이를 해경하는 방법으로 .copy() 메소드의 사용법을 분석해 본다.

동작

함수
    영상 어레이를 복사하여 반환하는 방법
        1) numpy.ndarray.copy(): 영상 어레이를 복사하여 새로운 어레이 abc를 반환한다.
        2) ndarray.astype('uint8'): ndarray를 uint8형으로 만들어 반환한다.
        3) ndarray.astype(dtype=np.uint8): ndarray를 uint8형으로 만들어 반환한다.

질문



"""
import cv2  as cv
import numpy as np

# ========================================================================================================
# 단계 0 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ========================================================================================================
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'd:/CV/Images/'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
Name2= 'colorbar_chart.jpg'
Name = 'lenna.bmp'
Name = 'monarch.bmp'
FullName = Path + Name
FullName2 = Path + Name2

# ========================================================================================================
# 단계 1 : 영상 파일을 읽어 들여 화면에 출력하기
# ========================================================================================================
image = cv.imread(FullName)      # default. IMREAD_COLOR라면 생략가능
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert image is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.

cv.imshow('step 1: image', image)
cv.waitKey(0)
cv.destroyWindow('step 1: image')

# ========================================================================================================
# 단계 2 : 영상 어레이를 원본을 바탕으로 생성하기. 복사본을 수정해도 원본은 수정되지 않는다.
# 영상 어레이를 복사하여 새로 만든다. image2는 image와 같은 내용으로 복사되어 생성된다.
# 아래 4가지 방법 중 어떤 것을 사용해도 문제를 일으키지 않는다.
# ========================================================================================================
image2 = image.copy()           #1. 복사하여 새로 만드는 가장 일반적인 방법
#image2 = image.astype('uint8')  #2. 원래는 데이터형을 바꾸는 목적인데 같은 형으로 복사하여 새로 만든다.
#image2 = image.astype(dtype=np.uint8)  #3. 2번과 같은 다른 표현 식
#image2 = image                  #4. 이 방법은 여기서는 문제를 일으키지 않지만 슬라이싱 동작에서 문제가 될 수 있다.

image2 = 255 - image2         # 복사본을 수정했다고 해서 원본(image)이 훼손되지 않는다.
print('step 2. 복사본을 수정했다고 해서 원본이 훼손되지 않는다.')
cv.imshow('step 2: image', image)
cv.imshow('step 2: image2', image2)
cv.waitKey(0)
cv.destroyAllWindows()


# ========================================================================================================
# 단계 3 : 영상 어레이를 포인터만 일치시켜 생성하기
# (원본 영상 image) vs (생성된 영상 image3) 관점의 변화를 관찰한다.
# assign에 의한 어레이의 생성은 포인터만 일치시킨다.
# 이 동작은 실제 영상 데이터 복사는 수행하지 않기 때문에 한 쪽만 수정하면 다른 한쪽도 함께 수정될 수 있다.
# ========================================================================================================

# 원본(image) 변형 없음. image3에 수정해서 넣는 과정에서 새로운 image3를 만들었다.
#image3 = image-50; print('step 3. 원본(image) 유지.. ')

# 원본(image) 변형 없음.
# 처음에는 image3가 image와 같은 데이터 공간을 공유했지만, image3에 수정해서 넣는 과정에서 새로운 image3를 만들었다.
#image3 = image; image3 -= 50; print('step 3. 원본(image) 유지.. ')

# 원본(image) 변형 발생. 복사본만 수정했는데 원본이 파괴된다.
#image3 = image; image3[:, :, :] = 0; print('step 3. 원본(image) 변형 발생.. ')
image3 = image; image[:, :, 0] = 0; print('step 3. 원본(image) 변형 발생.. ')

cv.imshow('step 3: image', image)
cv.imshow('step 3: image3', image3)
cv.waitKey(0)
cv.destroyAllWindows()     # This does not work in 2.7. Fine in 3.7


