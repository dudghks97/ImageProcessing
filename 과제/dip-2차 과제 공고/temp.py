import numpy as np
import cv2 as cv
import time

# HE
def Historgram_Equalization(img, weight, dtype):
    imgG = cv.cvtColor(img, cv.COLOR_BGR2GRAY)      # 그레이 스케일로 변환

    # 히스토그램(분산함수, DF) 를 구함
    hist, bins = np.histogram(a=imgG, bins=256, range=[0, 255])
    cdf = hist.cumsum()  # 누적분포함수

    # 매핑 후 LUT(Look Up Table) 생성
    mapping = cdf * 255 / cdf[255]
    LUT = mapping.astype(dtype)

    # LUT 기반 HE 시행, 컬러
    imgCeq = LUT[img]

    # 원본영상 반영비율 조절
    w = weight * 0.01
    imgCeq = np.clip((255 * (w * imgCeq / 255)) + (255 * ((1 - w) * img / 255)), 0, 255).astype('uint8')

    return imgCeq

# UM
def Unsharp_Masking(img, sigma, scale, dtype):
    img = img / 255  # 프레임 화소 값 정규화
    k = sigma * 6 + 1  # 커널의 크기
    blur = cv.GaussianBlur(src=img, ksize=(k, k), sigmaX=sigma)  # 블러링 영상
    um = img + scale * (img - blur)
    um = np.clip(um * 255, 0, 255).astype(dtype)

    return um
