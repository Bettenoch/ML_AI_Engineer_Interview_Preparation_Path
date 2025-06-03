fruits = ["apple", "banana", "cheryy", "oats"]
nutrients = [2, 9, 1, 3]

for i, fruit in enumerate(fruits):
    if fruit == "banana":
        fruits[i] = "laments"
    print(f"{i} : {fruit}")
print (fruits)

print (dict(zip(fruits, nutrients)))

def JoinTwo(arr1, arr2):
    result = {}  
    for i in range(len(arr1)):
        result[arr1[i]] = arr2[i]
    return result
        