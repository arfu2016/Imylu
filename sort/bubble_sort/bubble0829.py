"""
@Project   : Imylu
@Module    : bubble0829.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/29/18 6:13 PM
@Desc      : 
"""


def bubble_sort(a_list):
    l = len(a_list)
    for i in range(1, l):
        j = l-1
        while j >= i:
            key = a_list[j]
            if a_list[j] < a_list[j-1]:
                a_list[j] = a_list[j-1]
                a_list[j-1] = key
            j -= 1

    return a_list


if __name__ == '__main__':
    print(bubble_sort([5, 3, -1, 0]))
