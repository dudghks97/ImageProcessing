"""
matplotlib 함수를 이용한 화면 출력에 관계된 여러가지 팁을 보인다.

1. 화면 나누어 쓰기: subplot()
2. 1채널 영상 출력의 주의 사항: colormap 개념
3. 영상 어레이 선언하고, 데이터 편집해보기


"""

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np

fileName = "stinkbug.png"
fileName = "lenna.tif"
#fileName = "monarch.jpg"


# ---------------------------------------------------------------------------------------------------
# 2단계: matplotlib 함수로 영상을 처리한다.
# ---------------------------------------------------------------------------------------------------

# matplotlib에서 제공하는 함수를 통해 영상을 img_plt 어레이로 읽어들인다.
img_plt = img.imread(fileName)
print(f'type={type(img_plt)}, ndim={img_plt.ndim}, shape={img_plt.shape}, dtype={img_plt.dtype}')

# figure(): 새로운 창을 생성. num은 창의 이름 혹은 번호(number)를 지정한다.
fig = plt.figure(figsize=(14, 8), num='Step 2: Test')
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.85)       # 상하좌우 여분 공백 제어하기

fig.patch.set_facecolor('green')       # 화면 바탕 색상을 정한다.
#fig.patch.set_facecolor('lightblue')       # 화면 바탕 색상을 정한다.
fig.patch.set_alpha(0.1)        # Alpha, transparency: 0이면 투명, 1이면 완전 불투명.

# suptitle(): 창의 하위 제목을 지정한다.
plt.suptitle("Step 2: Simple Image Processing Examples\nClose this window to quit", fontsize=14, fontweight='bold')


plt.subplot(231)            # 2x3 배열의 subplt 창에서 1번째 창 지정
plt.imshow(img_plt)
plt.title('original')       # 해당 창의 타이틀
plt.axis("off")
# 1번 창에는 (x, y) 축의 픽셀 좌표가 출력됨.

plt.subplot(2, 3, 2)        # 2 x 3 배열의 subplt 창에서 2번째 창을 지정하는 다른 방법
rev = 255 - img_plt         # 영상 데이터는 0~255의 범위. 이 경우 반전(역상) 영상이 만들어진다.
plt.imshow(rev)
plt.title('reverse')
plt.xticks([]), plt.yticks([])
# 2번 창에는 (x, y) 축의 픽셀 좌표가 출력되지는 않으나 테두리는 남아 있음.

plt.subplot(233)
data = np.zeros((120, 180, 3), dtype=np.uint8)          # slicing <-> indexing


#for i in range(0, 40):
#    data[i, :, 0] = 255
data[0:40, :, 0] = 255      # Red. 0~39 행, : for all 열, R채널
data[40:80, :, 1] = 255      # Green
data[80:, :, 2] = 255      # Blue
plt.imshow(data)            # 3채널 영상
plt.title('Color in RGB(3ch.)')
plt.axis('off')        # 영상의 경계가 보이지 않는다.
#plt.xticks([]), plt.yticks([])       # 영상의 경계가 보인다.

plt.subplot(234)
data = np.zeros((120, 180), dtype=np.uint8)
data[50:100, :] = 255

# 다음 명령어의 주석을 해제하면 어떤 결과가 나오는지 관찰 바란다.
#plt.gray()     # figure 내의 모든 서브창에 대해 gray colormap을 쓰게 만든다.

#plt.imshow(data, 'viridis')                    # default colormap(viridis)을 사용.
plt.imshow(data, 'gray')
plt.colorbar()                      # 사용한 colormap을 bar 그림으로 함께 보여준다.

# false color => pseudo color. 의사색상

#plt.imshow(data, cmap='gray')      # gray colorma을 사용
plt.title('B/W BOX in 1ch')
#plt.axis('off')        # 영상의 경계가 보이지 않아 불편
plt.xticks([]), plt.yticks([])      # 영상의 경계가 보인다.

plt.subplot(235)        # red channel의 값만 보인다. 그런데 color map은 지정하지 않았다.
#plt.imshow(img_plt[:, :, 0])        # cmap= false color, pseudo color. 미지정
#plt.imshow(img_plt[..., 0])     # 위의 표현은 이렇게 바꾸어 쓸 수도 있다.
plt.imshow(img_plt[..., 0], cmap='jet')     # 큰 값은 뜨거운 색, 작은 값은 차가운 색으로 표현한다.
plt.colorbar()                      # 사용한 colormap을 bar 그림으로 함께 보여준다.

plt.title('1 ch: channel 0, Red')       # RGB 배열
plt.xticks([]), plt.yticks([])      # 영상의 경계가 보인다.

plt.subplot(236)
plt.imshow(img_plt[:, :, 0], cmap='gray')       # 흑백 계조 칼라맵을 사용한다.
plt.colorbar()                      # 사용한 colormap을 bar 그림으로 함께 보여준다.
plt.title('1 ch: channel 0, red')
plt.xticks([]), plt.yticks([])
print("현재 열린 창을 닫아야 종료합니다.....")
plt.show()

