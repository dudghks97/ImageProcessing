"""
1. 프로그램의 기능
    시그모이드 함수의 3개 파라미터에 대한 함수의 응답 값을 그래프로 표현한다.

2. 프로그램의 목표
    1) 시그모이드 변환의 특성(장점과 한계)을 이해한다.
    2) 시그모이드 함수의 3개 파라미터(m, w, E)의 제어 특성을 이해한다.

3. 프로그램에서 점검 포인트

4. 미션

5. 질문

"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

"""
# 실습 0 ======================================================================================
# 시그모이드 함수 그리기(1개)
import sys
eps = sys.float_info.epsilon
print('epsilon=', eps)

# m = mean value. 0~1. normally 0.5.
# w = weight.  0(no strength)~1(max strength)
# E = slope. weak ~ strong.


def sigmoid(m=0.5, w=0.5, E=8):
    print(m, w, E)
    r = np.arange(0, 256) / 255.0
    s = (w / (1 + (m/(r+eps))**E)) + (1-w) * r
    return r, s


mid_point = 0.5
slope = 8
for (weight, color) in [(0.0, 'b'), (0.2, 'g'), (0.4, 'r'), (0.8, 'c'), (1.0, 'm')]:
    r, s = sigmoid(w=weight, m=mid_point, E=slope)
    #plt.plot((r*255).astype(np.uint8), (s*255).astype(np.uint8), label='w='+str(weight))
    #plt.plot(r, s, label='w=' + str(weight))
    plt.plot(r, s, label=f'w={weight}', linewidth=0.8)


plt.xlabel(r'$\mathit{input: r}$', fontsize=15)
plt.ylabel(r'$\mathit{output: S(r)}$', fontsize=15)

plt.axis([0, 1, 0, 1])     # x_min, x_max, y_min, y_max.  plt.axis([0, 255, 0, 255])
plt.grid(color='b', linestyle='--', linewidth=0.5)
plt.legend(fontsize=15)
csfont = {'fontname':'Comic Sans MS'}
plt.title(f'Sigmoid Function: m={mid_point}, E={slope}', fontsize=15, **csfont)



plt.show()

exit(0)
"""


# 실습 1 ======================================================================================
# 시그모이드 함수 그리기(4개)
import sys
eps = sys.float_info.epsilon
print('epsilon=', eps)


# m = mean value. 0~1. normally 0.5.
# w = weight.  0(no strength)~1(max strength)
# E = slope. weak ~ strong.

def sigmoid(m=0.5, w=0.5, E=8):
    #print(m, w, E)
    r = np.arange(0, 256) / 255.0
    s = (w / (1 + (m/(r+eps))**E)) + (1-w) * r
    return r, s

def draw_sigmoid(sub_num, mid_point, slope):
    plt.subplot(sub_num)
    for (weight, color) in [(0.0, 'b'), (0.2, 'g'), (0.4, 'r'), (0.8, 'c'), (1.0, 'm')]:
    #for (weight, color) in [(0.1, 'b')]:
        r, s = sigmoid(w=weight, m=mid_point, E=slope)
        #plt.plot((r*255).astype(np.uint8), (s*255).astype(np.uint8), label='w='+str(weight))
        #plt.plot(r, s, label='w=' + str(weight))
        plt.plot(r, s, label=f'w={weight}', linewidth=0.8)


    plt.xlabel(r'$\mathit{input: r}$', fontsize=15)
    plt.ylabel(r'$\mathit{output: S(r)}$', fontsize=15)

    #plt.axis([0, 1, 0, 1])     # x_min, x_max, y_min, y_max.  plt.axis([0, 255, 0, 255])
    plt.grid(color='b', linestyle='--', linewidth=0.5)
    plt.legend(fontsize=13)
    csfont = {'fontname':'Comic Sans MS'}
    plt.title(f'Sigmoid Function: m={mid_point}, E={slope}', fontsize=15, **csfont)

fig = plt.figure(num='Sigmoid Functions', figsize=(12, 8))
plt.suptitle(f"Sigmoid Functions according to parameters", fontsize=18, fontweight='bold')
#fig.patch.set_facecolor('silver')  # 화면 바탕 색상을 정한다.
#fig.patch.set_alpha(0.3)

#ms_lst = [ [0.7, 20]]       # RuntimeWarning: overflow encountered in power
ms_lst = [ [0.7, 19]]       # 경고 오류 발생 안함.
#ms_lst = [[0.5, 8], [0.5, 19], [0.6, 8], [0.7, 19]]
ms_lst = [[0.5, 8], [0.5, 20], [0.6, 8], [0.3, 20]]
for i, (mid, slope) in enumerate(ms_lst):
    draw_sigmoid(221+i, mid, slope)

# sub 화면의 간격을 조정하려면 일단 출력하고 나서 진행해야 함.
# 아래 조정 작업 중에서 나중에 쓴 것이 설정을 overwrite함.
plt.tight_layout()  # 가로, 세로 관점에서 빈틈을 줄여 그림의 크기를 키움
#plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)  # 상하좌우 여백의 크기를 지정
#plt.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)  # 상하좌우 여백의 크기를 지정

plt.show()
