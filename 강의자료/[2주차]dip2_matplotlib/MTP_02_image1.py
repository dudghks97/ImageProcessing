"""
영상 파일 읽어들이기/보이기/저장하기

참조 링크:
    간략화된 numpy, matplotlib 및 TTS 등 한국어 문서 링크: https://wikidocs.net/14569
    matplotlib 부분 추천할 만함. 빠른 개념 파악을 위해..

"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image       # pillow image library
import cv2 as cv


fileName = "stinkbug.png"
fileName = "lenna.tif"
fileName = "monarch.jpg"

#"""
# 화소 : pixel(pixel element)  RGB: 3채널, numpy입장은 3차원, gray: 1채널(0~255)
# ---------------------------------------------------------------------------------------------------
#
# 참조: Matplolib Image Tutorial
#       https://matplotlib.org/stable/tutorials/introductory/images.html
#
# ---------------------------------------------------------------------------------------------------

# 아래 함수 1), 2), 4)번은 모두 np.ndarray 형으로 읽어낸다. 파일을 읽어 img 어레이 변수에 넣는다.
# 1), 2)번은 읽어낸 어레이를 수정할 수 없다.
#   이유: img.flags.owndata = False, img2.flags.writeable = False
#   대처방안: img = img.copy() 혹은 img = np.array(img)를 수행하면 flag가 바뀐다.
# 3), 4)번은 읽어낸 어레이를 수정할 수 있다.
# 4)번은 영상데이터가 BGR 순으로 배열되어 있다. 1), 2), 3)은 모두 BGR 배열
#   대처 방안: img = img[..., ::-1] 3차원의 순서를 역으로 재배열한다. cv.imshow()는 BGR 배열을 가정한다.

#img = plt.imread(fileName)             # 1) matplotlib.pyplot.read()
#img = mpimg.imread(fileName)           # 2) matplotlib.image.read()
#img = np.array(Image.open(fileName))   # 3) PIL.Image.imread(), pillow image library
img = cv.imread(fileName)[..., ::-1]    # 4) OpenCV library
#img = cv.imread(fileName)
# ===> 위의 처리는 우측과 같음: img = cv.imread(fileName); img = img[..., ::-1]

print(f"type(img)={type(img)}, img.shape={img.shape}, img.dtype={img.dtype}")
# type(img)=<class 'numpy.ndarray'>, img.shape=(512, 768, 3), img.dtype=uint8
# 영상은 보통 배열의 RGB 채널(3차원)로 읽어들이며, 데이터는 부호없는(unsigned integer) 8비트형이다. 0~255, 0~0xff
plt.imshow(img)                     # 화면에 출력한다. 처음에는 plt.figure() 안해도 새 창이 열림
plt.title('with coordinates')       # 기본적으로 좌표축의 눈금과 값이 나타난다.

new_name = 'tmp_' + fileName[:-3]+'jpg'      # 끝에서 3번째까지를 제외. 같은 이름의 새로운 파일 확장자 jpg 생성
plt.imsave(new_name, img)           # 영상을 새 이름으로 저장. png 파일로 저장하면 4채널로 늘어날 수도 있음

plt.figure()                        # 새 창을 연다
img2 = plt.imread(new_name)         # 저장한 파일을 읽어본다.
print(f"type(img2)={type(img2)}, img2.shape={img2.shape}, img2.dtype={img2.dtype}")
plt.imshow(img2)
plt.axis('off')                     # 눈금을 보이지 않게 한다. 테두리도 없앤다.
plt.title('with no boundary')

plt.figure()                        # 새 창을 연다
plt.xticks([]), plt.yticks([])      # 눈금을 보이지 않게 한다. 그러나 테두리는 남는다.
plt.title('with boundary')
plt.imshow(img2)

plt.show()

