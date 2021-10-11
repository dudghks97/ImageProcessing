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

# ---------------------------------------------------------------------------------------
# 실습 6: squeeze()함수, advanced topic
# squeeze() 함수는 어레이의 디멘전을 가장 최소화하여 표현할 수 있는 수준으로 줄인다.
# ---------------------------------------------------------------------------------------


print("실습 6: squeeze() 함수")
a0 = np.array([[1, 2, 3], [4, 5, 6]])       # 가장 필수적인 정보: (2, 3)
# a0.shape=(2, 3), a0=
# [[1 2 3]
#  [4 5 6]]
# after squeezing: sq.shape(2, 3)

a1 = np.array([[[1, 2, 3], [4, 5, 6]]])     # 가장 필수적인 정보: (2, 3)
# a1.shape=(1, 2, 3), a1=
# [[[1 2 3]
#   [4 5 6]]]
# after squeezing: sq.shape(2, 3)

a2 = np.array([[[[1], [4]]]])               # 가장 필수적인 정보: (2,)
# a2.shape=(1, 1, 2, 1), a2=
# [[[[1]
#    [4]]]]
# after squeezing: sq.shape(2,)

a3 = np.array([[[[1, 1]]]])                 # 가장 필수적인 정보: (2,)
# a3.shape=(1, 1, 1, 2), a3=
# [[[[1 1]]]]
# after squeezing: sq.shape(2,)

a4 = np.array([[[[5.0]]]])                  # 가장 필수적인 정보: (). 스칼라
# a4.shape=(1, 1, 1, 1), a4=
# [[[[5.]]]]
# after squeezing: sq.shape()
print("===  ", np.squeeze(a4), type(np.squeeze(a4)), 100+np.squeeze(a4))    # ===   5.0 <class 'numpy.ndarray'> 105.0


a5 = np.array([[3.0]])                      # 가장 필수적인 정보: (). 스칼라
for i, v in enumerate([a0, a1, a2, a3, a4, a5]):    # enumerate()는 인덱스, i와 값(v)를 반환한다.
    print(f'\na{i}.shape={v.shape}, a{i}=\n{v}')
    sq = np.squeeze(v)
    print(f'after squeezing: sq.shape{sq.shape}')

print("\nsqueeze details of last data, a5 ----- '")
print_att('sq')



