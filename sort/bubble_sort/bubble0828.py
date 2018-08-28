"""
@Project   : Imylu
@Module    : bubble0828.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/28/18 5:57 PM
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
    test_list = [5, 6, 3.2, 0, 9]
    print('The sorted list:', bubble_sort(test_list))
    test_list = [0, 1, 2, 3]
    print('The sorted list:', bubble_sort(test_list))
    test_list = [10, 9, 9, 3]
    print('The sorted list:', bubble_sort(test_list))
