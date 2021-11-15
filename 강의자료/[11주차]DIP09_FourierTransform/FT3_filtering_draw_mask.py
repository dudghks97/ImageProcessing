"""

주파수 공간에서 제거할 마스크를 그리는 연습용 프로그램

이 프로그램의 의도
    1. 코딩관점: 특정 원 내의 값을 0으로 만든다.
        이 프로그램(FT3_filtering_draw_mask)은 다른 프로그램에서 원을 정의하는 모듈로도 활용된다.
    2. 이론적 관점: 푸리에 변환 결과(스펙트럼으로 대표) * mask 가 어떤 의미를 가질까?
        1) 곱함으로써 0이 되는 부분은 해당 주파수 성분이 0이 되었다는 것이다.
        2) 곱한 결과를 역변환하면 특정 성분이 사라진 영상을 얻을 수 있다.
        3) 이는 필터링을 주파수 영역에서 할 수 있다는 말이다.
        4) 개념만을 보여준 것일 뿐. 실제 코딩은 fft2shift가 수반되어야 한다.

"""
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

"""
value = 3
scope = range(-value, value+1)
for i in scope:
    print(i)
exit(0)
"""




# diameter
def get_filter_mask(mask_size, d=20, option='circle', sigma=5):
    rows, cols = mask_size
    mask = np.ones((rows, cols), dtype=float)

    crow = rows // 2    # center of rows
    ccol = cols // 2

    # 1) 정사각형 마스킹: 중앙부 fft 값을 0에 가까운 값으로 만든다.
    if option == 'square':
        mask[crow-d:crow+d+1, ccol-d:ccol+d+1] = 0

    # 2) 원형 마스킹: 중앙부를 중심으로 반경 d인 원의 영역 내부를 0에 가까운 값으로 만든다.
    if option == 'circle':
        for x in range(-d, d+1):
            for y in range(-d, d+1):
                if x**2 + y **2 < d**2:
                    mask[crow+y, ccol+x] = 0.00000001

    if sigma > 0:
        mask_blr = cv.GaussianBlur(mask, (21, 21), 5)

    return mask_blr, mask



# 이 프로그램이 메인 자격으로 실행될 때만 아래 루틴이 수행된다.
# 만약 이프로그램이 모듈로 로드된다면 아래 루틴은 수행되지 않는다.
if __name__ == '__main__':
    cols = 512
    rows = 512


    plt.figure(num='filter mask design')

    # 1) 가상의 푸리에 변환 결과를 난수로 만들어 본다.
    # 0~255 범위의 난수로 채워진 (rows, cols) 어레이 반환
    img = np.random.uniform(0, 255, size=(rows, cols)).astype(np.uint8)
    print(img.dtype)

    plt.subplot(131)
    plt.imshow(img, cmap='gray')
    plt.title('(a) Fourier Result')
    #plt.axis('off')
    plt.xticks([]), plt.yticks([])


    # 2) mask를 생성한다.
    mask_blr, mask = get_filter_mask((rows, cols), d=50, option='circle', sigma=5)

    # 3) mask with no blurring
    plt.subplot(232)    # 블러되지 않은 마스크를 보인다.
    plt.imshow(mask, cmap='gray')
    plt.title('(b) mask(filter)')
    plt.xticks([]), plt.yticks([])

    img_masked = img/255 * mask # 마스크를 푸리에 변환 결과에 적용
    plt.subplot(233)
    plt.imshow(img_masked, cmap='gray')
    plt.title('(c) filtered result')
    plt.xticks([]), plt.yticks([])


    # 4) mask with blurring
    # blurring 처리된 마스크. ringing effect 감소 효과가 있다.
    plt.subplot(235)    # 블러 마스크를 보인다.
    plt.imshow(mask_blr, cmap='gray')
    plt.title('(d) mask_blr(filter)')
    plt.xticks([]), plt.yticks([])

    img_masked = img/255 * mask_blr     # 주파수 상의 필터링
    plt.subplot(236)
    plt.imshow(img_masked, cmap='gray')
    plt.title('(e) blur filtered result')
    plt.xticks([]), plt.yticks([])

    plt.show()

