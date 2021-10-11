"""

****** 주의사항 *****************
    1. plt.imread(), mpimg.imread()로 읽어서 영상 어레이를 반환받으면 수정 불가능한 어레이로 읽힌다.
        쓰기 금지 오류 발생: ValueError: assignment destination is read-only
        사례: img[:50, ...] = 0  # 0~49까지의 row에 대해 모든 column, channel에 대해 0을 넣는다.
    2. 다음과 같은 명령이 numpy 버전이 높아지면서 더 이상 지원되지 않는다.
        img.flags.owndata = True
        img.setflags(write=True)
"""


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image       # pillow image library

fileName = "lenna.tif"
#fileName = "monarch.jpg"
#fileName = "stinkbug.png"


img = plt.imread(fileName)             # 1) matplotlib.pyplot.read()
#img = mpimg.imread(fileName)           # 2) matplotlib.image.read()

print(f"type(img)={type(img)}, img.shape={img.shape}, img.dtype={img.dtype}")
# type(img)=<class 'numpy.ndarray'>, img.shape=(512, 768, 3), img.dtype=uint8

#img = img[0:50, ...]            # 이 동작은 새로운 어레이를 선언하는 것이므로 이상없음

# 아래 표현식을 주석 해제 처리하면 오류가 해결된다.
img = img.copy()                    # 오류 해결. 이렇게 하면 어레이가 쓸 수 있는 속성으로 바뀜

# 아래 동작은 오류 발생.
#img[:50, ...] = 0                   # ValueError: assignment destination is read-only

# 참고로 img 어레이의 flag 정보를 관찰해 보자.
print(f"img.flags ----------\n{img.flags}")      # owndata, writeable가 모두 False이다.
# numpy 1.16.0 이후로 owndata, writeable에 쓸 수 없음

# numpy version이 높아지면서 아래 동작은 지원하지 않고, 오류를 유발한다.
#img.flags.owndata = True
#img.setflags(write=True)

