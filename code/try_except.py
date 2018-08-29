"""
@Project   : Imylu
@Module    : try_except.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/28/18 6:38 PM
@Desc      : 
"""


def divide(x, y):
    try:
        result = x / y
    except ZeroDivisionError:
        print("division by zero!")
    else:
        print("result is", result)
    finally:
        print("executing finally clause")


if __name__ == '__main__':
    divide(10, 2)
    divide(5, 0)
