"""
numpy 배열 조작에 관계된 함수 혹은 메서드
NumPy Link : https://docs.scipy.org/doc/numpy/index.html

    1. arange: 연속적인 원소로 이루어진 1차원 배열을 반환한다.
    2. reshape: shape를 지정한 대로 바꾸어 반환한다. 데이터를 원본과 공유한다.
    3. resize: 원본 자체의 모양을 바꾼다. 반환 값이 없다.
    4. ravel: 임의의 배열을 1차원 배열로 만들어 반환한다. 데이터를 원본과 공유한다.
    5. flatten: 임의의 배열을 1차원 배열로 만들어 반환한다. ravel과는 달리 데이터를 원본과 공유하지 않는다.
    6. squeeze: 어레이의 디멘전을 가장 최소화하여 표현할 수 있는 수준으로 줄인다.

"""

import numpy as np



# ---------------------------------------------------------------------------------------------
# print_att()
#   입력 파라미터 변수(ndarray)에 대한 정보를 출력하는 함수
#   입력 파라미터는 변수의 스트링을 취한다.
# ---------------------------------------------------------------------------------------------
def print_att(str_a):

    # str_a를 스트링으로 간주하고 실제 값을 evaluate 한다.
    a = eval(str_a)
    print(f"{str_a}={a}")

    # type(): 자료형을 반환한다.
    print(f"type({str_a})={type(a)}")

    # shape: 행렬의 모양(행x열)을 지정하는 멤버변수
    print(f"{str_a}.shape={a.shape}")

    # ndim: 행렬의 차원을 지정하는 멤버변수
    print(f"{str_a}.ndim={a.ndim}")

    # dtype: 행렬의 원소의 데이터형을 지정하는 멤버변수. 행렬은 모두 같은 데이터형으로 구성된다.
    print(f"{str_a}.dtype={a.dtype}")

    # 한 원소를 표현하기 위해 필요한 데이터의 바이트 수
    print(f"{str_a}.itemsize={a.itemsize}")

    # array에 들어 있는 총 원소의 수
    print(f"{str_a}.size={a.size}")


print("\nrange() vs arange, 파이썬 내장함수  vs 넘파이 지원함수")
#--------------------------------------------------------------------------
#
#    a = range(start, end, increment)   예: range(1, 7, 2)
#    a = range(start, end) # increment 1
#    a = range(count)  0부터 1씩 증가시킨 값을 반환한다.
#
#    a = np.arange(start, end, increment)   예: arange(1.1, 7.9, 2.5), arange(1.1, 7.9) # increment 1
#    a = np.arange(start, end) # increment 1
#    a = np.arange(count)  0부터 1씩 증사시킨 값을 반환한다.       
#
#   두 함수의 비교
#                   파이썬 내장 range()                  numpy.arrange()
#   start, end      정수형만 지원                        부동소수도 지원
#   increment       부동소수 증분은 안됨                  부동소수 증분 가능
#   반환 값          range 객체(iterable object)         np.ndarray
#--------------------------------------------------------------------------
a = range(1, 10, 2)
print(f'a={a}, type(a)={type(a)}')

a = np.arange(1, 10, 0.5)       # start, stop, step
a = np.arange(1.1, 7, 1.01)
a = np.arange(5)
print(f'a={a}, type(a)={type(a)}, a.shape={a.shape}')
