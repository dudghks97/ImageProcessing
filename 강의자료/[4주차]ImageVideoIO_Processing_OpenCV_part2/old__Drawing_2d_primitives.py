"""
개요
    OpenCV에서 제공하는 다양한 2D primitive의 사용법을 제시한다.
    본 프로그램은 모든 실험 내용이 한 화면에 한꺼번에 실행된다.
    랜덤함수를 사용하여 출력 결과가 수행할 때마다 달라질 수 있다.

2. 주요 함수에 대한 소개 - Drawing Functions
    https://docs.opencv.org/4.0.1/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    OpenCV에서 제공하는 다양한 도형 그리기 함수를 소개한다.

3. 참고 사항 - 아래 예제를 다시 씀.
    1장- 06 Drawing 2d primitives_ markers, lines, ellipses, rectangles and text.py

"""

import numpy as np
import random
import cv2
import cv2 as cv

import numpy as np, random, cv2, cv2 as cv      # 위 4줄을 1줄로 임포트 가능.



#---------------------------------------------------------------------------------------------------------------
# 영상이 존재하는 폴더와 파일 이름을 지정하기.
#---------------------------------------------------------------------------------------------------------------
Path = 'd:\Work\StudyImages\Images\\'       # \\ 오류 발생 방지. \만 쓰면 오류.
Path = 'd:/Work/StudyImages/Images/'       # 이런 방식으로 사용 가능
#Path = '../../Images/'               # 현재 상위 폴더의 상위 폴더 아래에 있는 Images 폴더.
Path = '../data/'
Name = 'RGBColors.JPG'
Name= 'colorbar_chart.jpg'
Name = 'lenna.bmp'
Name = 'monarch.bmp'
FullName = Path + Name

#---------------------------------------------------------------------------------------------------------------
# 섹션 1: 영상 파일 열기
#---------------------------------------------------------------------------------------------------------------
# ImreadMode: 영상 데이터의 반환 모드를 결정
#   IMREAD_COLOR = 1            # default. 모노 영상도 3채널(RGB) 영상이 된다.
#   IMREAD_GRAYSCALE = 0        # 칼라 영상도 모노로 변환하여 연다. 1채널 영상이 됨.
#   IMREAD_UNCHANGED = -1       # 있는 그대로 열기.
image = cv2.imread(FullName, 1)
image_org = image.copy()        # 원본 영상을 저장해 둔다.

assert image is not None, 'No image file....!'  # 입력 영상을 제대로 읽어오지 못하여 NULL을 반환.
w, h = image.shape[1], image.shape[0]
print('image.shape=', image.shape)



#---------------------------------------------------------------------------------------------------------------
# 실습 1: circle 그리기
# 임의 좌표는 rand_pt()로 랜덤 생성하고, 그 좌표점을 중심으로 지정한 반지름, 색상으로 원을 그려 넣는다.
# https://docs.opencv.org/4.1.1/d6/d6e/group__imgproc__draw.html#gaf10604b069374903dbd0f0488cb43670
# void cv::circle (InputOutputArray 	img,                # prototype
#   Point center,           // Center of the circle, (x, y)
#   int radius,             // Radius of the circle.
#   const Scalar & color,   // Circle color. (B, G, R)
#   int thickness = 1,      // Thickness of the circle outline, if positive.
#                           // Negative values, like FILLED, mean that a filled circle is to be drawn.
#   int lineType = LINE_8,  // Type of the circle boundary. See LineTypes
#                           // LINE_4: 4-connected line, LINE_8: 8-connected line
#   int shift = 0)          // Number of fractional bits in the coordinates of the center and in the radius value.
#---------------------------------------------------------------------------------------------------------------
# 키입력을 할 때마다 4개의 원을 차례로 그린다.
x = 100; y= 300; center = (x, y); radius = 100; color = (255, 0, 0)     # (B, G, R) Blue, thickness=1
print(f'1a. Circle: center={center}, radius={radius}, color={color}')
cv2.circle(image,            # InputOutputArray
           center,           # Point center, (x, y)
           radius,           # int 	radius
           color)           # const Scalar &color, (B, G, R)
cv2.imshow("result", image); cv.waitKey()

x = 100; y= 300; center = (x, y); radius = 20; color = (0, 255, 0); thickness = cv2.FILLED      # Green. filled.
print(f'1b. Circle: center={center}, radius={radius}, color={color}, thickness={thickness}')
cv2.circle(image, center, radius, color, thickness)
cv2.imshow("result", image); cv.waitKey()

x = 100; y= 300; center = (x, y); radius = 80; color = (0, 0, 255); thickness = 4
print(f'1c. Circle: center={center}, radius={radius}, color={color}, thickness={thickness}')
cv2.circle(image, center, radius, color, thickness)
cv2.imshow("result", image); cv.waitKey()

x = 300; y= 100; center = (x, y); radius = 40; color = (255, 0, 255); thickness = 8
print(f'1d. Circle: center={center}, radius={radius}, color={color}, thickness={thickness}')
cv2.circle(image, center, radius, color, thickness,
           cv2.LINE_AA)     # antialiased line.
cv2.imshow("result", image); cv.waitKey()

#---------------------------------------------------------------------------------------------------------------
# 실습 2: 선 그리기
# void cv::line	( InputOutputArray 	img,
#   Point pt1,              // First point of the line segment.
#   Point pt2,              // Second point of the line segment.
#   const Scalar & 	color,  // Line color.
#   int thickness = 1,      // Line thickness.
#   int lineType = LINE_8,
#   int shift = 0)
#---------------------------------------------------------------------------------------------------------------

image = image_org.copy()        # 원본 영상에서 가져온다.
pt1 = (100, 10); pt2 = (100, 300); color = (0, 255, 0); thickness = 4
print(f'2a. Line: pt1={pt1}, pt2={pt2}, color={color}, thickness={thickness}')
cv2.line(image, pt1, pt2, color, 4)
cv2.imshow("result", image); cv.waitKey()

pt1 = (100, 300); pt2 = (300, 300); color = (255, 255, 255); thickness = 4
print(f'2b. Line: pt1={pt1}, pt2={pt2}, color={color}, thickness={thickness}')
cv2.line(image, pt1, pt2, color, 4)
cv2.imshow("result", image); cv.waitKey()


#---------------------------------------------------------------------------------------------------------------
# 실습 3: 화살표 선 그리기
# void cv::arrowedLine	(	InputOutputArray 	img,
#   Point pt1,              // First point of the line segment.
#   Point pt2,              // Second point of the line segment.
#   const Scalar & 	color,  // Line color.
#   int thickness = 1,      // Line thickness.
#   int lineType = LINE_8,
#   int shift = 0,
#   double 	tipLength = 0.1)    // The length of the arrow tip in relation to the arrow length
#---------------------------------------------------------------------------------------------------------------

#image = image_org.copy()        # 원본 영상에서 가져온다.
pt1 = (100, 10); pt2 = (100, 300); color = (0, 255, 255); thickness = 4
print(f'3a. Arrow: pt1={pt1}, pt2={pt2}, color={color}, thickness={thickness}')
cv2.arrowedLine(image, pt1, pt2, color, thickness, cv2.LINE_AA)
cv2.imshow("result", image); cv2.waitKey(0)

pt1 = (100, 300); pt2 = (300, 300); color = (255, 255, 0); thickness = 4
print(f'3b. Arrow: pt1={pt1}, pt2={pt2}, color={color}, thickness={thickness}')
cv2.arrowedLine(image, pt1, pt2, color, thickness, cv2.LINE_AA)
cv2.imshow("result", image); cv2.waitKey(0)


#---------------------------------------------------------------------------------------------------------------
# 실습 4: 사각형 그리기
# void cv::rectangle	(	InputOutputArray 	img,
#   Point 	pt1,            // Vertex of the rectangle.
#   Point 	pt2,            // Vertex of the rectangle opposite to pt1 .
#   const Scalar & 	color,  // Rectangle color or brightness (grayscale image).
#   int thickness = 1,      // Thickness of the circle outline, if positive.
#                           // Negative values, like FILLED, mean that a filled circle is to be drawn.
#   int lineType = LINE_8,  // Type of the circle boundary. See LineTypes
#                           // LINE_4: 4-connected line, LINE_8: 8-connected line
#   int shift = 0)          // Number of fractional bits in the coordinates of the center and in the radius value.
#---------------------------------------------------------------------------------------------------------------
image = image_org.copy()
pt1 = (100, 10); pt2 = (300, 300); color = (0, 255, 255); thickness = 4
print(f'4. Rectangle: pt1={pt1}, pt2={pt2}, color={color}, thickness={thickness}')
cv2.rectangle(image, pt1, pt2, color, thickness)
cv2.imshow("result", image)
cv2.waitKey(0)


#---------------------------------------------------------------------------------------------------------------
# 실습 5: 타원형 그리기
# https://docs.opencv.org/4.1.1/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
# void cv::ellipse(	InputOutputArray 	img,
#   Point center,               // Center of the ellipse.
#   Size 	axes,               // Half of the size of the ellipse main axes. (width, height)
#   double 	angle,              // Ellipse rotation angle in degrees.
#   double 	startAngle,         // Starting angle of the elliptic arc in degrees.
#   double 	endAngle,           // Ending angle of the elliptic arc in degrees.
#   const Scalar & 	color,      // Ellipse color.
#   int thickness = 1,      // Thickness of the circle outline, if positive.
#                           // Negative values, like FILLED, mean that a filled circle is to be drawn.
#   int lineType = LINE_8,  // Type of the circle boundary. See LineTypes
#                           // LINE_4: 4-connected line, LINE_8: 8-connected line
#   int shift = 0)          // Number of fractional bits in the coordinates of the center and in the radius value.
#---------------------------------------------------------------------------------------------------------------

image = image_org.copy()        # 원본 영상에서 가져온다.

x = 200; y= 300; center = (x, y)
width, height = 200, 100; axes = (width, height)
#width, height = 100, 200; axes = (width, height)
angle = 0; startAngle = 0; endAngle = 360;     # 45도 기울어지진 타원형 전체
color = (255, 255, 255); thick = 3
print(f'5a. Ellipse: center={center}, (width, height)={axes}, angle={angle}, start={startAngle}, end={endAngle}, color={color}, thickness={thick}')
cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thick)
cv2.imshow("result", image); cv2.waitKey(0)


x = 200; y= 150; center = (x, y)
width, height = 200, 100; axes = (width, height)
angle = 0; startAngle = 90; endAngle = 270;     # 45도 기울어지진 타원형 일부
color = (0, 255, 255); thick = 4
print(f'5b. Ellipse: center={center}, (width, height)={axes}, angle={angle}, start={startAngle}, end={endAngle}, color={color}, thickness={thick}')
cv2.ellipse(image, center, axes, angle, startAngle, endAngle, color, thick)
cv2.imshow("result", image); cv2.waitKey(0)

#cv2.ellipse(image, rand_pt(), rand_pt(0.3), random.randrange(360), 0, 360, (255, 255, 255), 3)


#---------------------------------------------------------------------------------------------------------------
# 실습 6: 글씨 넣기
# void cv::putText(	InputOutputArray 	img,
#   const String & 	text,               // Text string to be drawn.
#   Point org,                          // Bottom-left corner of the text string in the image.
#   int fontFace,                       // Font type, see HersheyFonts.
#   double fontScale,                   // Font scale factor that is multiplied by the font-specific base size.
#   Scalar color,                       // Text color.
#   int thickness = 1,                  // Thickness of the lines used to draw a text.
#   int lineType = LINE_8,              // Line type. See LineTypes
#   bool bottomLeftOrigin = false)      // When true, the image data origin is at the bottom-left corner.
#                                           Otherwise, it is at the top-left corner.
#---------------------------------------------------------------------------------------------------------------

print(f'6. Text..,')
image = image_org.copy()        # 원본 영상에서 가져온다.
x = 200; y= 300; org = (x, y)       # Bottom-left corner
cv2.putText(image, 'OpenCV', org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
cv2.putText(image, 'OpenCV', (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 3, cv2.LINE_AA)
cv2.imshow("result", image)
cv2.waitKey(0)




#---------------------------------------------------------------------------------------------------------------
# 실습 7: 랜덤 변수로 랜덤 위치에 랜덤 칼라, 랜덤 크그로 글씨 쓰기
#---------------------------------------------------------------------------------------------------------------

print(f'7. Text: random location & color..,')
# import random             # 파이썬 자체에 내장된 모듈.
# a = random.randrange(5)       # 0~4까지의 임의의 정수를 반환.
# a = random.randrange(1, 7)    # 1~6까지의 임의의 정수를 반환.
# a = random()                  # 0부터 1 사이의 임의의 float

# 전역변수 w, h의 범위에 있는 2개의 정수 값을 반환하는 함수.
# 영상(x, h)의 가로 x 세로 영역 안에 있는 랜덤 (x,y) 좌표로 활용예정
def rand_pt(mult=1):      # 본 함수는 (width, height) 내의 임의의 난수 좌표를 생성한다.
    return (random.randrange(int(w*mult)),          # 함수 내에서 w, h는 전역변수를 그대로 활용한다.
            random.randrange(int(h*mult)))

image = image_org.copy()        # 원본 영상에서 가져온다.

for i in range(100):
    cv2.putText(image, 'OpenCV', rand_pt(), cv2.FONT_HERSHEY_SIMPLEX,
                random.randrange(1, 3),    # font scale 1~2까지. 1 or 2
                (random.randrange(256), random.randrange(256), random.randrange(256)),  # 0~255
                random.randrange(1, 3))     # thickness 1~2
                #2)
cv2.imshow("result", image)
cv2.waitKey(0)

