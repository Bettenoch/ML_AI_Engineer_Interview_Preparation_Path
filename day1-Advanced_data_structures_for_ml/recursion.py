def factorial(n):
    if n == 1 or n == 0:
        return 1
    else:
        return n * factorial(n-1)
    
print (factorial(4))

arr = [3 , 6, 9, 12, 9, 78]

print (arr[:2])

def binarySearch(arr, low, high, val):
    while low <= high:
        mid = (high + low) // 2
        if arr[mid] == val:
            return mid
        elif val < arr[mid]:
            high = mid -1
        elif val > arr[mid]:
            low = mid + 1
    return -1


if __name__ == '__main__':
    arr = [3, 8, 0, 12, 44, 23, 9]
    val = 12
    
    arr.sort()
    result = binarySearch(arr, 0, len(arr) - 1, val)
    
    if result == -1:
        print(f"Element {val} not found")
    else:
        print("Element is found", arr[result])
        
#Time Complexity: O(log N)
#Auxiliary Space: O(1)

def binarySearchRecursive(arr, low, high, val):
    if low <= high:
        mid = (low + high) // 2
        
        if arr[mid] == val:
            return mid
        elif val < arr[mid]:
            return binarySearchRecursive(arr, low, mid - 1, val)
        else:
            return binarySearchRecursive(arr, mid + 1, high, val)
    
    else: 
        return -1
    
    
arr34= [2, 5, 67, 23, 0, 12, 33]

arr34.sort()
print(arr34)

low = 0
high = len(arr34) - 1
val = 12
found = binarySearchRecursive(arr = arr34, low = low, high = high, val = val)
print (arr34[found])
