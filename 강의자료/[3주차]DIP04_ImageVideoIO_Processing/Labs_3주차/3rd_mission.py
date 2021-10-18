import os
import cv2 as cv

fname = '../data/colorbar_chart.jpg'

# 0. fname으로 지정된 파일을 화면에 출력
img = cv.imread(fname)
cv.imshow(fname, img)
cv.waitKey()

# 1. 가로 및 세로 정보를 변수 a, b 에 넣기
print(img.shape)
a = img.shape[0]
b = img.shape[1]
print(f'가로 = {a}, 세로 = {b}')

# 2. 영상의 면적 출력
area = a * b
print(f'면적 = {area}')

# 3. 칼라 / 모노 영상 여부 검사
channel = img.shape[2]

if channel == 3:
    print('컬러 영상 입니다!')
elif channel == 1:
    print('모노 영상 입니다!')
else:
    print('컬러, 모노 둘 다 아닙니다!')

# 4. 영상 화면에 출력, 파일 이름이 타이틀 바에 출력 되어야 함.
fbasename = os.path.basename(fname)
cv.imshow(fbasename, img)

# 5. 영상 quality를 지정하여 현재의 폴더에 jpg 영상으로 저장
q = input("Quality is : ")
quality = int(q)

cv.imwrite('tmp.jpg', img, (cv.IMWRITE_JPEG_QUALITY, quality))
img = cv.imread('tmp.jpg')

# 6. 저장된 영상을 읽어 화면에 출력 타이틀 바의 이름 : qaulity = x
cv.imshow(f'quality = {quality}', img)
cv.waitKey()

# ============================================================================ #
# 영상 품질을 키보드로 입력 받아 저장하는 프로그램 작성
# 확장자 : jpg
input_fname = '../data/lenna.tif'
out_fname = 'abc.jpg'

img = cv.imread(input_fname)
print(f'The original file is : {os.path.basename(input_fname)}')

quality = int(input("Type Integer Number(1 ~ 100) : "))

cv.imwrite(out_fname, img, (cv.IMWRITE_JPEG_QUALITY, quality))
print(f'The out file is : {out_fname}')

out_img = cv.imread(out_fname)

cv.imshow(os.path.basename(input_fname), img)
cv.imshow(out_fname, out_img)
cv.waitKey()

# ============================================================================ #
