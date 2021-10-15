"""
1. 프로그램의 기능 - 감마 변환에 사용되는 변환함수의 그래프를 출력한다.
    감마 <1, 감마 > 1 두가지 사례에 대해 입력과 출력의 관계를 도시한다.

2. 프로그램의 목표
    1) 감마값에 따른 감마 변환의 특성을 이해한다.
    2) 임의의 1차 신호 함수를 출력하는 일반화된 루틴을 설계한다.

3. 프로그램에서 점검 포인트
    1) 1차 신호 함수를 파이썬 루틴으로 정의한다.
    2) 함수를 원하는 형태로 출력하는 method를 익힌다.
        - legend, grid, axis, xlable, ylabel

4. 미션
    matplotlib.pyplot에서 지원하는 메소드의 기법을 활용하여 아래의 내용을 변경하시오.
    1) xlabel(), ylabel()을 변경하여 출력된 문자의 크기를 최대로 확대하여 출력하시오.
    2) 2개 그래프의 타이틀의 문자를 확대하고 색상을 바꾸어 출력하시오.
        힌트: https://matplotlib.org의 검색창에 matplotlib.pyplot을 입력하면 종속 함수 리스트를 볼 수 있고,
             matplotlib.pyplot.xlabel을 입력하면 이중 xlabel 함수의 사용법을 볼 수 있다.

5. 질문

"""

"""
# 실습 1 ======================================================================================
# gamma 변환 결과 2개를 도시하고 그 특성을 살핀다.
# 점검 포인트
#   1) 감마가 1보다 작을 때와 클 때 변환 결과가 달라지는 양상에 주의한다.
#   2) 그래프의 표현법을 잘 숙지하도록 하자.
# 참고: Writing mathematical expressions
# https://matplotlib.org/tutorials/text/mathtext.html

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

gamma = 0.3     #-------------- 그래프 1: gamma < 1
r = np.arange(0, 255+1, 1, np.float)
print('1: r.shape=', r.shape, 'r.dtype=', r.dtype)

y = 255 * (r / 255.0) ** gamma
print('2: y.shape=', y.shape, 'y.dtype=', y.dtype)

y = y.astype('uint8')
print('3: y.shape=', y.shape, 'y.dtype=', y.dtype)

plt.plot(r, y, 'r', label=r'$\gamma=$'+str(gamma))

# {}안의 스트링을 italic 서체로 label를 작성한다.
plt.xlabel(r'$\mathit{input: p}$', fontsize=12)
plt.ylabel(r'$\mathit{output: }$'+'$p^\gamma $', fontsize=12)

# 아래 3개 함수는 x, y의 범위를 지정하는 역할을 수행한다.
#plt.xlim(left=0, right=255); plt.ylim(bottom=0, top=255)
plt.axis([0, 255, 0, 255])     # x_min, x_max, y_min, y_max
#plt.xticks([0, 255]); plt.yticks([0, 255])     # grid 없어짐.

plt.grid(color='b', linestyle='--', linewidth=0.5)  # ()안에 인자가 있으면 grid(True)로 간주하여 그리드 출력.


gamma=3         #-------------- 그래프 2: gamma > 1
plt.plot(r, (255 * (r / 255.0) ** gamma).astype('uint8'), 'g', label=r'$\gamma=$'+str(gamma))
plt.legend()

csfont = {'fontname':'Comic Sans MS'}
plt.title('Gamma Transformation Function', fontsize=15, **csfont)

plt.show()
exit(0)
#=============================================================================================
"""



# 실습 2 ======================================================================================
# 다양한 gamma 변환 결과를 도시한다.
# 점검 포인트
#   감마 변환 함수를 정의하고 이를 이용한다.

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def gammafunction(r, gamma):
    y = 255 * (r/255.0) ** gamma
    return y.astype('uint8')

r = np.arange(0, 255+1, 1, np.uint8)
print('1: r.shape=', r.shape, 'r.dtype=', r.dtype)

def open_figure(win_name='', fig_size=(12, 8), sub_title='', font_size=10):
    fig = plt.figure(num=win_name, figsize=fig_size)
    if sub_title == '':
        sub_title = win_name
    plt.suptitle(sub_title, fontsize=font_size, fontweight='bold')
    #fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
    #fig.patch.set_alpha(0.3)

    # 원한다면 창을 최대화하여 그림을 출력한다.
    #figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()
    return fig

open_figure(win_name='Gamma Functions according to parameters', fig_size=(12, 8),
            sub_title="", font_size=18)


plt.subplot(121)
# 오류!! color를 숫자로 표현하면 레전드를 2배로 만들어 내는 것으로 판명되었다.
# 아래 루틴에서 color은 문자열로 표시했어야 함. 레전드를 만들지 않으면 오류가 나타나지 않음.
# 일부 인터넷 예제에 이런 사례가 활용되었었다.
#for (gamma, color) in [(0.2, 1), (0.3, 2), (0.4, 3), (0.5, 4), (0.6, 5)]:

# 교정 1!! 칼라를 숫자를 사용하지 않고 일일이 지정하였다.
#for (gamma, color) in [(0.2, 'b'), (0.3, 'g'), (0.4, 'r'), (0.5, 'c'), (0.6, 'm')]:
#    plt.plot(r, gammafunction(r, gamma), color, label='gamma='+str(gamma))

# 교정 2!! 색상을 랜덤 자동 지정에 맡겨 둔다.
for gamma in [0.2, 0.3, 0.4, 0.5, 0.6]:
    plt.plot(r, gammafunction(r, gamma), label=r'$\gamma=$'+str(gamma))

# 다음 2개의 정보가 본 사례에는 적용되지 않는 것으로 확인되었으나, 그대로 남겨둔다.
# 1) \mathit{} - forces the math italic mode. ---> plplot에서는 작동안되는 듯 하다.
#       {}안의 스트링을 italic 서체로 바꾼다. 주의: print()는 안됨.
#       pyplot에서는 '$...$'로 문자열 ...을 $로 둘러싸서 지정하면 이탤릭체로 출력된다. mathit 작동 안함.
#       https://www.tutorialspoint.com/tex_commands/mathit.htm
# 2) r string: raw string. \ 문자를 escape 문자가 아닌 그냥 단순한 문자(literal character)로 처리한다.
#        print(r'Hi\nHello') => 수행하면 출력되는 문자열 =>  Hi\nHello
#       https://www.journaldev.com/23598/python-raw-string
#       * 실제로는 '\gamma' 혹은 r'\rgamma' 모두 그리스문자 변환이 이루어진다.
#
#       결론 =>pyplot()에서는 $...$로 이태릭으로 변환하고. 이 안에서는 \로 그리스 문자 변환을 할 수 있다.

#plt.xlabel(r'\mathit{input: p}', fontsize=15)       # 이태릭체로 바뀌지 않는다. mathit 문자도 출력됨
plt.xlabel('$input: p$', fontsize=18)      # 이태릭체로 바뀐다.
plt.ylabel(r'$output: $' + '$p^\gamma $', fontsize=18)   # gamma가 그리스 문자로 바뀐다.

plt.axis([0, 255, 0, 255])     # x_min, x_max, y_min, y_max
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.legend(fontsize=15)

csfont = {'fontname':'Comic Sans MS'}           # 폰트를 지정해 보았다.
#plt.title(r'Gamma Transformation Function($\gamma$<1)', fontsize=15, **csfont)     # 이것도 되고,
plt.title('Gamma Transformation Function($\gamma$<1)', fontsize=15, **csfont)       # 이것도 된다.
plt.subplot(122)

# 정상!! 색상을 랜덤 자동 지정에 맡겨 둔다.
for gamma in [2, 3, 4, 5, 6]:
    plt.plot(r, gammafunction(r, gamma), label='$\gamma=$'+str(gamma))

#plt.xlabel('input'); plt.ylabel('output')
# {}안의 스트링을 italic 서체로 label를 작성한다.
plt.xlabel(r'$\mathit{input: p}$', fontsize=18)
plt.ylabel(r'$\mathit{output: }$'+'$p^\gamma $', fontsize=18)

plt.axis([0, 255, 0, 255])     # x_min, x_max, y_min, y_max
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.legend(fontsize=15)
plt.title(r'Gamma Transformation Function($\gamma$>1)', fontsize=15, **csfont)

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
#plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # 상하좌우 여백의 크기를 지정
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # 상하좌우 여백의 크기를 지정


plt.show()
exit(0)

#=============================================================================================



"""
#===============================================
# 연습(1)
import matplotlib.pyplot as plt

y1 = [1, 2, 3]
y2 = [3, 2, 1]
yy = [y1, y2]

a1, = plt.plot(yy[0], label='Line0')
a2, = plt.plot(yy[1], label='Line1')

h=list()        # empty list
h.append(a1)    # append hanlde
h.append(a2)

plt.legend(handles=h)
plt.show(); exit(0)
#===============================================
"""

"""
#===============================================
# 연습(2)
import matplotlib.pyplot as plt

y1 = [1, 2, 3]
y2 = [3, 2, 1]
yy = [y1, y2]
h=list()        # empty list

for i in yy:
    #print(i)
    a, = plt.plot(i, label='Line'+str(i))
    h.append(a)     # append hanlde

plt.legend(handles=h)
plt.show(); exit(0)
#===============================================
"""

"""
#===============================================
# 연습(3)
import matplotlib.pyplot as plt

y1 = [1, 2, 3]
y2 = [3, 2, 1]
yy = [y1, y2]
h=list()        # empty list

# Create plots with pre-defined labels.
fig, ax = plt.subplots()

for i in yy:
    #print(i)
    a, = ax.plot(i, label='Line'+str(i))
    h.append(a)     # append hanlde

plt.legend(handles=h)
plt.show(); exit(0)
#===============================================
"""

"""
#------------------------------------------------------------------------------------------------------
# 연습(4)
# matplotlib.pyplot.legend
# suplots에서 반환하는 핸들의 메소드를 이용한 legend() 활용
#------------------------------------------------------------------------------------------------------
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html?highlight=matplotlib%20pyplot%20legend#matplotlib.pyplot.legend
import numpy as np
import matplotlib.pyplot as plt


# Make some fake data.
a = b = np.arange(0, 3, .5); print(b)
c = np.exp(a)
d = c[::-1]    # c를 거꾸로 배열

# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'r:', label='Data length')
ax.plot(a, c + d, 'g', label='Total message length')

#legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
#ax.legend()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
print(len(handles))
print(len(labels))

# Put a nicer background color on the legend.
#legend.get_frame().set_facecolor('C4')

plt.show()
exit(0)
#------------------------------------------------------------------------------------------------------
"""
