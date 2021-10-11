"""
# pyplot.plot() 함수로 그래프 그리기
# pyplot.plot(x, y, fmt, kwargs) 파라미터 제어기법을 보인다.

# https://matplotlib.org/  -> 여기서 plot을 검색하여 적당한 것을 찾는다.
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html?highlight=plot#matplotlib.pyplot.plot


참조 링크:
    간략화된 numpy, matplotlib 및 TTS 등 한국어 문서 링크: https://wikidocs.net/14569
    matplotlib 부분 추천할 만함. 빠른 개념 파악을 위해..

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


#"""
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
plt.figure()
plt.plot(x, y2)
plt.grid('on')          # grid를 표시한다. plt.grid()도 'on'과 같은 효과
#plt.show()              # 실제 그림들은 이 명령에 의해 화면에 출력된다. 창을 닫아야 다음 줄로 넘어간다.


# fig. 3 ------ 여기서는 color, line, marker를 지정하여 출력해 본다.
plt.figure()        # 새로운 창을 연다. 맨 처음 여는 창은 이 명령어 없어도 열린다.
fmt = 'go-'         # 데이터가 있는 부분을 'g'(green) 색상의 'o'로 표시한다. 선의 모습은 '-'이다.
y3 = [18, 16, 14, 12, 0]
plt.plot(x, y3, fmt)
plt.plot(x, y2, '^b--')     # 삼각형 마커, blue, 점선,
plt.grid(False)     # grid를 제거한다.

# 현재까지는 메모리에만 그래프를 그려 넣는다.

plt.show()          # 실제 그림들은 이 명령에 의해 화면에 출력된다. Non Interactive mode(default)의 특징.
# show()순간에서 대기하고 있음. 다음으로 넘어가려면 두 개의 창을 모두 닫아야 한다.
print("Windows were closed...")

