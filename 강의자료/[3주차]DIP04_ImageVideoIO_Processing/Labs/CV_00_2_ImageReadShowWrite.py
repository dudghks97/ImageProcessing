"""

개요
    영상 파일을 읽어들여 화면에 출력하고 저장한다.
    저장할 때 칼라 영상을 모노 그레이로 저장하거나, 압축 품질, 파일 확장자(jpg, png)를 바꾸어 저장한다.

동작
    단계 1: 영상 파일의 영상 읽기. 다음 함수를 통해 지정된 영상을 읽어내어 영상 데이터를 numpy.ndarray 객체로 반환한다.
    단계 2: 영상 데이터 화면에 출력하기.
    단계 3: 칼라 영상을 모노그레이로 변환하여 저장한 후, 다시 읽어내어 채널 정보가 잘 유지되는지 비교하여 확인한다.
    단계 4: 칼라 영상에 대해 압축 품질을 지정하여 영상을 다른 품질 혹은 다른 파일 형식으로 저장한 후,
            다시 읽어서 화면에 출력하여 영상의 품질을 비교한다.

질문
    1. JPEG 파일로 저장할 때 화질을 지정하여 저장하는 방법은?
    2. 영상 파일을 읽을 때 칼라 파일을 1채널의 모노 파일로 읽어 들이는 방법은?
    3. 1채널의 모노 영상을 3채널의 칼라 데이터로 읽어 들이는 방법은?

"""

import cv2 as cv

# ----------------------------------------------------------------------------------------------------------------------
# 섹션 1 :  영상이 존재하는 폴더와 파일 이름을 지정하기.
# ----------------------------------------------------------------------------------------------------------------------
#Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
#Path = 'd:/Work/StudyImages/Images/'
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'       # . 현재 폴더, .. 바로 윗 폴더, ../data/ => 현재 폴더 바로 위 폴더 아래에 있는 data 폴더 아래
Name = 'RGBColors.JPG'
Name2= 'colorbar_chart.jpg'
Name = 'lenna.tiff'
Name = 'monarch.bmp'
FullName = Path + Name
#FullName2 = Path + Name2

# ----------------------------------------------------------------------------------------------------------------------
# 단계 1: 영상 파일의 영상 읽기. 다음 함수를 통해 지정된 영상을 읽어내어 영상 데이터를 numpy.ndarray 객체로 반환한다.
# cv.imread(파일, flags)에서; -1=있는 그대로 읽기. 0=모노(1채널)로 읽기, 1=칼라(3채널)로 읽기(default)
# ----------------------------------------------------------------------------------------------------------------------
#flags = cv.IMREAD_COLOR                    # 3채널로 읽는다. default. =1
imgC = cv.imread(FullName, 1)         # ==> 같음. flags = cv.IMREAD_COLOR
# assert condition, message  : condition이 false이면  message 출력하면서 AssertError 발생.
assert imgC is not None, f'No image file [{Name}] ....!'  # 입력 영상을 제대로 읽어오지 못하여 None를 반환.

imgM = cv.imread(FullName, 0)      # IMREAD_GRAYSCALE = 0. Read in mono.

print(f"imgC.shape={imgC.shape}, imgM.shape={imgM.shape}")


# ----------------------------------------------------------------------------------------------------------------------
# 단계 2: 영상 데이터 화면에 출력하기.
# 아래 함수들은 cv.waitKey() 함수가 있어야 화면에 내용물을 출력한다.
# ----------------------------------------------------------------------------------------------------------------------
winname_img ="'imgC': ImReadMode=" + str(cv.IMREAD_COLOR)     # =1. default. BGR color. 3 channel.
cv.imshow(winname_img, imgC)
winname_imgM = "'imgM': ImReadMode=" + str(cv.IMREAD_GRAYSCALE)  # =0. mono gray. 1 channel.
cv.imshow(winname_imgM, imgM)
cv.waitKey()       # 이곳에서 키보드 입력을 기다린다.
cv.destroyAllWindows()      # 창을 모두 닫는다.



fname = Name[0:-4]      # 단계 3, 4에서 쓰임. 파일 이름에서 확장자(.bmp, .jpg, .png 등 점 포함)를 제거한다.
# ----------------------------------------------------------------------------------------------------------------------
# 단계 3: 칼라 영상을 모노그레이로 변환하여 저장한 후, 다시 읽어내어 채널 정보가 잘 유지되는지 비교하여 확인한다.
# ----------------------------------------------------------------------------------------------------------------------

# 모노 그레이로 변환하여 저장한다.
imgM = cv.cvtColor(imgC, cv.COLOR_BGR2GRAY)     # 영상의 칼라 좌표 변환함수. 참고: cv.COLOR_GRAY2BGR
cv.imshow(f"3) 'imgM' from 'cvtColor()': imgM.shape={imgM.shape}", imgM)
file_mono = fname + '_mono'+'.jpg'      # 저장할 파일의 이름과 확장자.
cv.imwrite(file_mono, imgM)               # 2차원 영상을 저장한다.

# 저장한 파일을 다시 읽어낸다.
img = cv.imread(file_mono)   # -1 : as is
cv.imshow(f"3) 'img' from '{file_mono}': img.shape={img.shape}", img)

# 구조체의 정보를 화면에 출력하여 비교한다.
print(f"\n3) array.shape check before writing into file: imgM.shape={imgM.shape}")
print(f"3) array.shape check after reading from file: img.shape={img.shape}")       # 모노 영상인데 왜 3차원일까?
print("왜 읽어내니까 3채널 영상이 되었나요? 쓰기 전과 같은 채널로 읽어 내려면?")
cv.waitKey(0)       # 이곳에서 키보드 입력을 기다린다.


# ----------------------------------------------------------------------------------------------------------------------
# 단계 4: 압축 품질을 지정하여 영상을 다른 품질 혹은 다른 파일 형식으로 저장한 후, 다시 읽어서 화면에 출력하여 영상의 품질을 지정한다.
# 압축을 사용하는 파일 형식은 영상 품질로 압축률을 통제할 수 있다.
# jpg와 png 형식의 파일로 저장한다. => 파일 이름의 확장자로 지정 가능하다.
# ----------------------------------------------------------------------------------------------------------------------

# 4.1) 압축 품질을 5로 지정하여 jpeg으로 저장한다. 저장한 파일을 다시 읽어서 화면에 출력하여 영상의 품질을 지정한다.
quality = 5              # quality : 1 to 100. The higher is the better. quality is 95 default.
file_qual = fname + f'(q={quality:#02d})'+'.jpg'              # 저장할 파일의 이름 = 원래이름+(q=05)
#여러 개의 파라미터를 제어할 때는 아래와 같이 파라미터 이름과 값을 나열하면 된다.
# 이들 파라미터는 리스트 혹은  튜플로서 한 개의 파라미터로 입력된다.
cv.imwrite(file_qual, imgC, (cv.IMWRITE_JPEG_QUALITY, quality))       # 영상을 주어진 품질(quality)로 저장한다.

print(f"\n4.1) 'imgC' was saved in '{file_qual}' with 'quality={quality}'.")
img = cv.imread(file_qual)
cv.imshow(f"4.1 '{file_qual}'", img)
cv.waitKey(0)       # 이곳에서 키보드 입력을 기다린다.

# 4.2) 압축 품질을 80으로 지정하여 jpeg으로 저장한다. 저장한 파일을 다시 읽어서 화면에 출력하여 영상의 품질을 지정한다.
quality = 80              # quality : 1 to 100. The higher is the better. quality is 95 default.
file_qual = fname + f'(q={quality:#02d})'+'.jpg'              # 저장할 파일의 이름
cv.imwrite(file_qual, imgC, (cv.IMWRITE_JPEG_QUALITY, quality))       # 영상을 주어진 품질(quality)로 저장한다.
print(f"\n4.2) 'imgC' was saved in '{file_qual}' with 'quality={quality}'.")
img = cv.imread(file_qual)
cv.imshow(f"4.2 '{file_qual}'", img)
cv.waitKey(0)       # 이곳에서 키보드 입력을 기다린다.

# 4.3) 압축 품질을 지정하여 png로 저장한다. default=1.
# png 파일 형식:
#   lossless compression을 사용하므로 화질의 차이를 느끼기 어려움.
#   JPG 파일보다 용량이 매우 크며, 품질을 달리 지정해도 파일의 크기의 차이도 별로 없다.
#   단지 압축 시간의 차이가 조금 있을 뿐...
quality = 1              # png 파일의 품질. 0~9로 지정. 높은 값(9)이 압축이 심하다고 하나 약간 줄어들며 대신 압축시간이 더 걸림.
file_png = f'{fname}(q={quality:#02d}).png'
cv.imwrite(file_png, imgC, (cv.IMWRITE_PNG_COMPRESSION, quality) )
print(f"\n4.3) 'imgC' was saved in '{file_png}' with quality={quality} PNG format.")
img = cv.imread(file_png)
cv.imshow(f"4.3 '{file_png}'", img)
cv.waitKey(0)       # 이곳에서 키보드 입력을 기다린다.
exit()




#====================================================================================================================
# 미션 과제(1)
# 변수 fname로 주어진 영상 파일에 대해 주어진 미션을 완수히시오.
# fname = "../data/colorbas_chart.jpg"      # 단, 파일의 실제 이름은 바뀔 수도 있음.
# 0. fname으로 지정된 파일을 화면에 출력하시오..
# 1. 가로(column, 열) 및 세로(row, 행) 정보를 변수 a, b에 넣으시오.
#     a.shape
#    이를 이용하여 영상의 가로, 세로 정보를 출력하시오. => print(f'가로={a}, 세로={b}')
# 2. 영상의 면적을 출력하시오. => 면적 = 23674834
# 3. 파일이 칼라인지 모노 영상인지를 검사하여 그 결과를 출력하시오. => 영상은 '칼라 혹은 모노'입니다.
# 4. 해당 영상을 화면에 출력하시오. 이때 파일 이름이 타이틀 바에 출력되어야 합니다. 경로(path)를 제외하면 가점
# 5. 읽은 영상을 quality를 지정하여 현재의 폴더에 JPG 영상으로 저장하시오. 파일 이름은 tmp.jpg
#    quality는 사용자의 입력을 받아 지정하게 하면 가점. 못하겠으면 5로 고정.
# 6. 저장된 영상을 읽어 화면에 출력하시오 이때 타이틀 바의 이름은 "quality=?"
#     ? 값은 5로 고정하거나, 사용자 입력으로 정한 값으로 정한다. 사용자 입력 가점.
#====================================================================================================================



#====================================================================================================================
# 미션 과제(2)
# 영상 품질을 키보드로 입력 받아 저장하는 프로그램을 작성하시오.
# 이때 파일 확장자는 jpg를 가정하며 이는 입력받지 않는다.
# 활용 예: lenna.tif를 입력받아 quality=20인 abc.jpg로 변환하여 저장하한다.
#   input file:lenna.tif
#   out file:abc
#   quality:20
#
#  만약 입력 영상이 위치한 곳이 현재 폴더 위의 data 폴더에 위치한다면
#  입력할 파일명은 ==> "../data/lenna.tif"
#====================================================================================================================

# input() 함수로 받은 데이터는 string 형이다.
# 따라서 정수가 필요한 자리에 사용할 때는 이를 int() 함수를 사용하여 정수로 바꾸어야 한다.
abc = input("Type integer number = ")
print(f"You typed {abc}, type(abc)={type(abc)}")

q = int(abc)
print(q + 10)       # abc+10은 불가능. 스트링과 정수를 더할 수 없음.


#====================================================================================================================
# 미션 과제(3)
# 파일의 이름과 영상 품질을 키보드로 입력 받아 저장하는 프로그램을 작성하시오.
# 사용 예: python lenna.bmp out.jpg 20         # lenna.bmp 영상을 입력 받아 out.jpg로 저장.
#====================================================================================================================


#====================================================================================================================
# 기록 보관용... 먼저 칼라 좌표계의 학습이 필요함.. => 나중에 다룸.
# 영상을 색차(chroma) 성분과 휘도(luminance) 성분으로 나누어 압축하는데 각각에 대한 압축품질을 개별로 설정할 수도 있다.
#q_chroma = 1; q_luma = 80
#file_qual2 = fname + f'_Chro={q_chroma:#02d}_Luma={q_luma:#02d}.jpg'  # 저장할 파일의 이름
#cv.imwrite(file_qual2, imgC, (cv.IMWRITE_JPEG_CHROMA_QUALITY, q_chroma, cv.IMWRITE_JPEG_LUMA_QUALITY, q_luma))
#print(f"3.2c) 'imgC' was saved in '{file_qual2}' with luminance quality={q_luma}, chroma quality={q_chroma}.")
#====================================================================================================================
