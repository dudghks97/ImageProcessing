import pickle

with open("d:/dip/data/encoding_and_label.bin", "rb") as file:
    all_member_face, all_member_encodings1d_lst, all_member1d_label_lst = pickle.load(file)

for i, (a_dsc, a_lbl) in enumerate(zip(all_member_encodings1d_lst, all_member1d_label_lst)):
    print(f"{i}: label={a_lbl}", end='')
    for k in range(10):  # 한 사진의 인코딩 값을 앞의 10개만 찍어본다.
        print(f"{a_dsc[k]:7.3f}", end=" ")
    print()