"""

1. OpenCV 영상 어레이(BGR)와 matplotlib 영상 어레이(RGB)
    칼라 채널 배열의 차이점 및 핸들링 방안
2. matplotlib 화면 출력 상의 interactive mode 운영법
    - OpenCV처럼 보이게 하기: 키를 입력하거나(OpenCV 경우) 마우스 클릭
3. 화면 전체를 저장하기
    -fig.savefig("result.png")

"""

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

fileName = "stinkbug.png"
fileName = "lenna.tif"
fileName = "monarch.jpg"

# ---------------------------------------------------------------------------------------------------
# 단계 3: OpenCV로 읽은 영상을 matplotlib.plt로 화면에 출력한다.
# ---------------------------------------------------------------------------------------------------

import cv2 as cv

# openCV에서 제공하는 함수를 통해 영상을 img_cv 어레이로 읽어들인다.
img_cv = cv.imread(fileName)    # BGR 순으로 배열..
print(f'type={type(img_cv)}, ndim={img_cv.ndim}, shape={img_cv.shape}, dtype={img_cv.dtype}')

# matplotlib에서 제공하는 함수를 통해 영상을 img_plt 어레이로 읽어들인다.
img_plt = img.imread(fileName)
print(f'type={type(img_plt)}, ndim={img_plt.ndim}, shape={img_plt.shape}, dtype={img_plt.dtype}')

fig = plt.figure(num=3)
fig.patch.set_facecolor('silver')       # 화면 바탕 색상을 정한다.

# suptitle(): 창의 하위 제목을 지정한다.
plt.suptitle("Step 3: OpenCV image handling", fontsize=14, fontweight='bold')

# OpenCV 함수로 읽은 영상 어레이를 출력한다.
plt.subplot(221)
plt.imshow(img_cv)              # 읽어들인 영상 배열은 BGR 순서인데, 출력하는 함수는 RGB라서 올바른 색상이 표현되지 않는다.
plt.title('img read by openCV(Bad!)')
plt.axis('off')

# OpenCV 함수로 읽은 영상 어레이를 출력하려면 BGR 배열을 RGB 배열로 바꾸어야 한다.
plt.subplot(222)
plt.imshow(img_cv[..., ::-1])     # 0~2번 채널 배열을 2~0번 채널로 재배열하여 출력한다. BGR -> RGB
plt.title('img read by openCV(Good!)')
plt.axis('off')

gray = cv.cvtColor(img_cv, cv.COLOR_BGR2GRAY)   # gray로 변환한다.
plt.subplot(223)
plt.imshow(gray)
plt.colorbar()
plt.title('gray with default cmap')
plt.axis('off')

plt.subplot(224)
plt.imshow(gray, cmap='gray')
plt.colorbar()
plt.title("gray with cmap='gray'")
plt.axis('off')

print("현재 열린 창을 닫아야 다음 루틴으로 진행합니다....")
plt.show()

# ---------------------------------------------------------------------------------------------------
# 단계 4: 화면을 꽉 채워서 출력하기. 화면 저장하기.
# ---------------------------------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 8), num='Step 4: 아무런 키나 입력하면 종료합니다.')
fig.patch.set_facecolor('blue')       # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.3)        # Alpha, transparency: 0이면 투명, 1이면 완전 불투명.

#plt.suptitle("pyplt Image, 아무런 키나 입력하면 종료합니다", fontsize=14, fontweight='bold')
plt.suptitle("pyplt Image, Press any key to quit", fontsize=14, fontweight='bold')

plt.subplot(131)
plt.imshow(img_plt)
plt.title('original')
#plt.title('원본 영상')         # 한글은 안되는군요....
plt.axis('off')

plt.subplot(1, 3, 2)
rev = 255 - img_plt
plt.imshow(rev)
plt.title('reverse')
plt.axis('off')

plt.subplot(133)
plt.imshow(img_plt[:, :, 0], cmap='gray')
plt.title('channel 0')
plt.xticks([]), plt.yticks([])

# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout() # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01) # 상하좌우 여백의 크기를 지정


# matplotlib에도 파일 저장함수가 있다. openCV는 imwrite(파일명, 어레이)
img.imsave('rev.jpg', rev)

# matplotlib는 영상 어레이뿐만 아니라 윈도 전체를 저장하는 기능을 지원한다.
fig = plt.gcf()
fig.savefig("result.png")
#fig.savefig("result.jpg")   # jpg는 안되는 듯. 오류 발생.

#plt.show()  # 이 함수를 사용하면 영상을 출력하고 창을 닫을 때 까지 대기한다.
# plt.show() 대신에 아래 함수를 쓰면 화면을 출력하고, 키입력 혹은 마우스 클릭을 대기한다.
print("아무런 키나 입력하면 종료합니다.")
plt.waitforbuttonpress()


# 참조 -----------------------
# plt.close()       # 현재 활성창을 닫는다.
# plt.close(fig)    # 핸들 'fig'로 창을 닫는다.
# plt.close(num)    # 숫자 'num'을 닫는다.
# plt.close(name)   # 이름이 'name'인 창을 닫는다.
# plt.close('all')    # 모든 창을 닫는다.


