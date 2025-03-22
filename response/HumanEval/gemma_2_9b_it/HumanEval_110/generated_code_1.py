def method():
    lst1 = [1, 3, 5, 2]
    lst2 = [4, 6, 8, 10]

    #  Count the number of odd numbers in lst1
    odd_count = lst1.count(i for i in lst1 if i % 2 != 0)

    # Check if the number of odd numbers in lst1 is less than or equal to the number of even numbers in lst2
    if odd_count <= len(lst2) - lst2.count(i for i in lst2 if i % 2 == 0):
        return "YES"
    else:
        return "NO"

output = method()
print(output)