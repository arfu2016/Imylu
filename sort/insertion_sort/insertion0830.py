"""
@Project   : Imylu
@Module    : insertion0830.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/30/18 6:19 PM
@Desc      : 
"""


def insertion_sort(a_list):
    """这里的实现是，从左到右，从小到大，所以是小的往前面插"""
    l = len(a_list)
    for i in range(1, l):
        key = a_list[i]
        j = i-1
        while a_list[j] > key and j >= 0:
            a_list[j+1] = a_list[j]
            a_list[j] = key
            j -= 1
    return a_list


if __name__ == "__main__":
    print(insertion_sort([-1, -2, 7, 0, 3]))
    print(insertion_sort([0, 1, 3, 5]))
    print(insertion_sort([10, 9, 9, 6]))
