def method():
    def count_nums(arr):
        def signed_digit_sum(n):
            # Convert the number to string, and calculate the sum of its digits
            s = str(n)
            total_sum = 0
            
            # The first digit will be negative if the number is negative
            if s[0] == '-':
                total_sum += int(s[1]) * -1  # First signed digit is negative
                digits = s[2:]  # Remaining digits are positive
            else:
                total_sum += int(s[0])  # First digit is positive
                digits = s[1:]  # Remaining digits
            
            # Sum the remaining digits
            total_sum += sum(int(d) for d in digits)
            return total_sum
        
        count = 0
        for num in arr:
            if signed_digit_sum(num) > 0:
                count += 1
        
        return count

    # Test case for validation
    test_array = [123, -123, 45, -60, -7, 0, 10]
    output = count_nums(test_array)
    
    return output

# Call the method and print the output
print(method())