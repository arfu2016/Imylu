"""
@Project   : Imylu
@Module    : quick0907.py
@Author    : Deco [deco@cubee.com]
@Created   : 9/7/18 11:57 AM
@Desc      : 
"""


def quick_sort(nums, low, high):
    if len(nums) == 0:
        return None
    if low >= high:
        return nums
    x = nums[high]
    i = low - 1
    for j in range(low, high):
        # j是小于等于x的值
        # i是小于等于x的值当中，最右边界
        if nums[j] <= x:
            i += 1
            nums[j], nums[i] = nums[i], nums[j]

    nums[i+1], nums[high] = nums[high], nums[i+1]

    quick_sort(nums, low, i)
    quick_sort(nums, i + 2, high)

    return nums


if __name__ == '__main__':
    a_list = [-8, 7, 6, 10, 28, 0, -29]
    length = len(a_list)
    print(quick_sort(a_list, 0, length-1))
