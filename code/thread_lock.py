"""
@Project   : Imylu
@Module    : thread_lock.py
@Author    : Deco [deco@cubee.com]
@Created   : 8/28/18 6:43 PM
@Desc      : 
"""
import threading
import inspect
import time


class Thread(threading.Thread):
    def __init__(self, t, *args):
        threading.Thread.__init__(self, target=t, args=args)
        self.start()


count = 0
lock = threading.Lock()


def incre():
    global count
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    print("Inside %s()" % caller)
    print("Acquiring lock")

    # lock.acquire()
    # print("Lock Acquired")
    # count += 1
    # time.sleep(2)
    # lock.release()
    # print("Lock Released")

    # lock.acquire()
    # print("Lock Acquired")
    # try:
    #     count += 1
    #     time.sleep(2)
    # finally:
    #     lock.release()
    #     print("Lock Released")

    with lock:
        print("Lock Acquired")
        count += 1
        time.sleep(2)


def bye():
    while count < 4:
        incre()
        print('Increased by bye()')


def hello_there():
    while count < 6:
        incre()
        print('Increased by hello_there()')


def main():
    hello = Thread(hello_there)
    goodbye = Thread(bye)
    # nonblocking call

    while count < 6:
        time.sleep(0.5)
        print(count)


if __name__ == '__main__':
    main()
