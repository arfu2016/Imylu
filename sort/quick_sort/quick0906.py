"""
@Project   : Imylu
@Module    : quick0906.py
@Author    : Deco [deco@cubee.com]
@Created   : 9/6/18 3:57 PM
@Desc      : 
"""
from collections import deque


def quick_sort(nums):
    if not nums:
        return None
    # 初始化队列并传入数组左右边界
    que = deque([0, len(nums) - 1])
    # 如果队列中有元素
    while que:
        # 获取队列中的一对左右边界
        low = que.popleft()
        high = que.popleft()
        # 如果下标越界则跳过
        if low >= high:
            continue
        # x作为标杆
        x = nums[high]
        # i作为所有小于标杆元素的右边界
        i = low - 1
        # 遍历
        # print('low:', low)
        # print('high:', high)
        for j in range(low, high):
            # 元素小于标杆
            if nums[j] <= x:
                # i右移到大于标杆元素的左边界
                i += 1
                # 交换大于标杆元素的左边界，和当前这个小于标杆的元素
                nums[i], nums[j] = nums[j], nums[i]
                # print(nums)
        # 标杆元素与大于标杆元素的左边界元素进行交换
        # print('i+1:', i+1)
        # print('high:', high)
        nums[i + 1], nums[high] = nums[high], nums[i + 1]
        # print(nums)
        # 取两个数组的左右边界写入队列
        # 注意如果元素都大于标杆那么i有可能小于low，反之i + 2也可能大于high
        # 所以要通过"if low >= high: continue"这句来控制数据越界的情况
        que.extend([low, i, i + 2, high])

    return nums


def quick_sort2(nums, low, high):
    if not nums:
        return None
    if low >= high:
        # nums的low到high部分是只有一个元素的列表?
        # 递归结束的条件：分的只剩一个元素
        return nums
    x = nums[high]
    # 最右边的元素作为标杆元素
    # i作为所有小于标杆元素的右边界
    i = low - 1
    # i从low - 1开始
    for j in range(low, high):
        # 元素小于标杆
        if nums[j] <= x:
            # j表征的是小于等于x的值
            # i右移到大于标杆元素的左边界
            i += 1
            # i表征的是大于x的值
            # 加1表示i和j是同步的
            # 交换大于标杆元素的左边界，和当前这个小于标杆的元素
            nums[i], nums[j] = nums[j], nums[i]
            print(nums)
    print('i+1:', i+1)
    print('high:', high)
    nums[i + 1], nums[high] = nums[high], nums[i + 1]
    # i是小于x的右边界，所以i+1必然大于x，把x换到中间来
    print(nums)
    print(i + 1)
    # if i + 1 == low:
    #     return nums[i + 1] + quick_sort2(nums, i+2, high)
    # if i + 1 == high:
    #     return quick_sort2(nums, low, i) + nums[i + 1]
    # return
    # quick_sort2(nums, low, i) + nums[i + 1] + quick_sort2(nums, i+2, high)
    quick_sort2(nums, low, i)
    quick_sort2(nums, i + 2, high)
    return nums


if __name__ == '__main__':
    print(quick_sort([1, -1, 6, -5, 2]))
    print(quick_sort2([1, -1, 6, -5, 2], low=0, high=4))
    print(quick_sort2([5, -4, 3, 2, 1, 9], low=0, high=5))
