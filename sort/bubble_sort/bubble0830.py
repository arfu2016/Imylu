"""
@Project   : Imylu
@Module    : bubble0830.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/30/18 1:21 PM
@Desc      : 
"""


def bubble_sort(a_list):
    l = len(a_list)
    j = l-1
    for i in range(1, l):
        while j >= i:
            key = a_list[j]
            if a_list[j] < a_list[j-1]:
                a_list[j] = a_list[j-1]
                a_list[j-1] = key
            j -= 1
    return a_list


if __name__ == '__main__':
    print(bubble_sort([0, -2, 9, 7, 10]))
