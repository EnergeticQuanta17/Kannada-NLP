import random

my_list = [random.randint(1, 100) for _ in range(80000)]

import time
start = time.perf_counter()
my_list.sort()
print(time.perf_counter()-start)

def binary_search(arr, target):
    left = 0
    right = len(arr) - 1

    while left <= right:
        mid = (left + right) // 2

        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False

start = time.perf_counter()
for i in range(1_000_000):
    binary_search(my_list, random.random())

print(time.perf_counter()-start)