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



#--------------------------------------------------------------------------
print("\nNumPy에서 제공하는 reshape 메서드를 활용한 연습")
# numpy.arange([start, ]stop, [step, ]dtype=None)
#   start부터 stop까지 step의 증분으로 순차적으로 증가/감소하는 1차원 배열을 반환한다. 데이터형은 dtype로 주어진다.
#   https://numpy.org/doc/stable/reference/generated/numpy.arange.html
#
# 
# numpy.reshape(a, newshape, order='C')
#   https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html?highlight=reshape#numpy.reshape
# 기능: 해당 어레이에 대해 shape가 다른 뷰(View)를 생성한다.
# 특징:
#   가능한 복사해서 데이터를 생성한다고 하지만 배열의 데이터를 공유할 수 있다.
#   즉, b = a.reshape((2,3))을 수행하고 난 후 a를 수정하면 b도 같이 수정될 수도 있다. 
#   반환한 데이터의 배열이 연속성이 없을 수도 있다.
#   order='C' : C-like index order
#   원본 배열의 데이터를 공유하지 않는 새로운 복사본을 만들고 싶으면 ndarray.copy 메서드를 사용한다.
#--------------------------------------------------------------------------
print()
a = np.arange(5)    # 0~4까지 1씩 증가하는 1차원 어레이 생성. 1차원, ndim= 1.
print_att('a')
# a=[0 1 2 3 4]
# type(a)=<class 'numpy.ndarray'>
# a.shape=(5,)
# a.ndim=1
# a.dtype=int32
# a.itemsize=4
# a.size=5

print()
b = np.arange(5).reshape(1, 5)     # row*column = 1*5. ndim= 2. a= [[0 1 2 3 4]]
print_att('b')
# b=[[0 1 2 3 4]]
# type(b)=<class 'numpy.ndarray'>
# b.shape=(1, 5)
# b.ndim=2
# b.dtype=int32
# b.itemsize=4
# b.size=5

print()
c = np.arange(15).reshape(3, 5)     # row*column = 3*5. ndim= 2
print_att('c')
# c=[[ 0  1  2  3  4]
#  [ 5  6  7  8  9]
#  [10 11 12 13 14]]
# type(c)=<class 'numpy.ndarray'>
# c.shape=(3, 5)
# c.ndim=2
# c.dtype=int32
# c.itemsize=4
# c.size=15
