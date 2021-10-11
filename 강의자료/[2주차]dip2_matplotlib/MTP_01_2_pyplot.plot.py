"""
# pyplot.plot() 함수로 그래프 그리기
# pyplot.plot(x, y, fmt, kwargs) 파라미터 제어기법을 보인다.

# https://matplotlib.org/  -> 여기서 plot을 검색하여 적당한 것을 찾는다.
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html?highlight=plot#matplotlib.pyplot.plot

colors
'b'	blue
'g'	green
'r'	red
'c'	cyan
'm'	magenta
'y'	yellow
'k'	black
'w'	white

line style
'-'	solid line style
'--'	dashed line style
'-.'	dash-dot line style
':'	dotted line style

Markers
'.'	point marker
','	pixel marker
'o'	circle marker
'v'	triangle_down marker
'^'	triangle_up marker
'<'	triangle_left marker
'>'	triangle_right marker
'1'	tri_down marker
'2'	tri_up marker
'3'	tri_left marker
'4'	tri_right marker
'8'	octagon marker
's'	square marker
'p'	pentagon marker
'P'	plus (filled) marker
'*'	star marker
'h'	hexagon1 marker
'H'	hexagon2 marker
'+'	plus marker
'x'	x marker
'X'	x (filled) marker
'D'	diamond marker
'd'	thin_diamond marker
'|'	vline marker
'_'	hline marker

"""

import matplotlib.pyplot as plt         # alias


"""
#---------------------------------------------------------------------------------
# 실습 1: (x, y) 평면에서 (x, y) 데이터 세트를 이용하여 그래프 그리기
# fig. 1: 선의 색상과 속성을 제어 안하고 그리기
# fig. 2: fmt 부분을 지정하여 선의 색상과 속성을 제어하기
#---------------------------------------------------------------------------------

# fig. 1 ------
x = [0, 1, 2, 3, 4]
y1 = [1, 3, 5, 7, 9]
y2 = [8, 6, 4, 2, 0]
plt.plot(x, y1)             # color, line, marker를 지정하지 않았음.
plt.plot(x, y2)
plt.grid('on')          # grid를 표시한다. plt.grid()도 'on'과 같은 효과
#plt.show()              # 실제 그림들은 이 명령에 의해 화면에 출력된다. 창을 닫아야 다음 줄로 넘어간다.


# fig. 2 ------ 여기서는 color, line, marker를 지정하여 출력해 본다.
plt.figure()        # 새로운 창을 연다. 맨 처음 여는 창은 이 명령어 없어도 열린다.
fmt = 'go-'         # 데이터가 있는 부분을 'g'(green) 색상의 'o'로 표시한다. 선의 모습은 '-'이다.
y3 = [18, 16, 14, 12, 0]
plt.plot(x, y3, fmt)
plt.plot(x, y2, '^b--')     # 삼각형 마커, blue, 점선,
plt.grid(False)     # grid를 제거한다.
plt.show()          # 실제 그림들은 이 명령에 의해 화면에 출력된다. Non Interactive mode(default)의 특징.
# show()순간에서 대기하고 있음. 다음으로 넘어가려면 두 개의 창을 모두 닫아야 한다.
print("Windows were closed...")
exit()
"""


#"""
#---------------------------------------------------------------------------------
# 실습 2: 1차 함수, 2차함수를 그려본다.
#---------------------------------------------------------------------------------
import numpy as np

#x = range(-10, 11, 1)
# arange() 함수는 start부터 stop까지 increment씩 증가시킨 numpy형 1차원 어레이를 만든다.
x = np.arange(-10.0, 10+1, 1)     # (시작점, 끝점+하나 더, 증분)
print(f'type(x)={type(x)}, x.shape={x.shape}, x.dtype={x.dtype}, x.ndim={x.ndim}')
print(f'start: x[0]={x[0]}, stop: x[-1]={x[-1]}')
# type(x)=<class 'numpy.ndarray'>, t.shape=(21,), t.dtype=int32
# start: x[0]=-10, stop: x[-1]=10


plt.figure(num=5)       # 타이틀 바를 지정하면서 새 창을 연다
plt.axhline(color='black', lw=1.5)          # 수평 중심선 긋기. 검은 색, 굵기 1.5
plt.axvline(color='black', lw=1.5)          # 수직 중심선 긋기. 검은 색, 굵기 1.5
plt.xlabel('x value')       # x 축에 라벨 적기
plt.ylabel('y value')       # y 축에 라벨 적기
plt.grid()                  # 모눈 종이처럼 그리드 그리기

a = 3       # 기울기
b = 20       # 절편. y 축과 만나는 지점
y1 = a * x + b
plt.plot(x, y1, 'c*-', label='y1=3x+20')

y2 = -4 * x - 20
plt.plot(x, y2, 'bo--', label='y2=-4x-20')
plt.legend()

plt.figure('2nd order equations')       # 타이틀 바를 지정하면서 새 창을 연다
plt.axhline(color='black', lw=1.5)          # 수평 중심선 긋기. 검은 색, 굵기 1.5
plt.axvline(color='black', lw=1.5)          # 수직 중심선 긋기. 검은 색, 굵기 1.5
plt.xlabel('x value')       # x 축에 라벨 적기
plt.ylabel('y value')       # y 축에 라벨 적기
#plt.grid(True)                  # 모눈 종이처럼 그리드 그리기
plt.grid(color='b', linestyle='--', linewidth=0.5)

y1 = x ** 2 + 10
plt.plot(x, y1, 'g+-', label='y1=x**2+10')

y2 = x ** 2 + 50
plt.plot(x, y2, 'rs-', label='y2=x**2+50')      # s: square marker

y3 = (x-2.5) ** 2 - 50
plt.plot(x, y3, 'y^-.', label='y3=(x-2.5)**2-50')        # y=yellow ^=삼각형 마커

plt.xlim(left=-10, right=10)      # x축의 최소, 최대값을 지정한다.
plt.ylim([-75, 150])    # =plt.ylim(top=150, bottom=-75)  # y축의 최소, 최대값을 지정한다.

plt.legend()
plt.show()
exit()

