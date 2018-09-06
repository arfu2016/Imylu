"""
@Project   : Imylu
@Module    : insertion0906.py
@Author    : Deco [deco@cubee.com]
@Created   : 9/6/18 2:59 PM
@Desc      : 
"""


def insertion_sort(a_list):
    l = len(a_list)
    for i in range(1, l):
        j = i-1
        key = a_list[i]
        while key < a_list[j] and j >= 0:
            a_list[j+1] = a_list[j]
            a_list[j] = key
            j -= 1
    return a_list


if __name__ == '__main__':
    print(insertion_sort([1, -1, 0, 5, 2]))
