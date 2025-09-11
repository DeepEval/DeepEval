class BigNumCalculator:
    @staticmethod
    def add(num1, num2):
        return str(int(num1) + int(num2))

    @staticmethod
    def subtract(num1, num2):
        return str(int(num1) - int(num2))

    @staticmethod
    def multiply(num1, num2):
        # Convert strings to lists of digits
        num1_digits = list(num1)
        num2_digits = list(num2)
        
        # Initialize the result with zeros
        result = ['0'] * (len(num1_digits) + len(num2_digits))
        
        # Perform multiplication digit by digit, adjusting for alignment
        for i in range(len(num1_digits)):
            for j in range(len(num2_digits)):
                # Multiply the corresponding digit in num1 and num2
                product = int(num1_digits[i]) * int(num2_digits[j])
                
                # Sum the products from the current position to the next
                position = i + j
                while position > 0:
                    product += result[position - 1]
                    result[position - 1] = str(product % 10)
                    result[position] = str(product // 10)
                    position -= 1
        
        # Trim trailing zeros from the result
        while result and result[0] == '0':
            result.pop(0)
        
        # Convert list of digits to a string and prepend the sign of num1
        sign_of_num1 = 1 if int(num1) >= 0 else -1
        result = [str(sign_of_num1 * int(digit)) for digit in result]
        return ''.join(result)

# Test cases
if __name__ == "__main__":
    bigNum = BigNumCalculator()
    print(bigNum.add("12345678901234567890", "98765432109876543210"))  # Output: '111111111011111111100'
    print(bigNum.subtract("12345678901234567890", "98765432109876543210"))  # Output: '-86419753208641975320'
    print(bigNum.multiply("12345678901234567890", "98765432109876543210"))  # Output: '1219326311370217952237463801111263526900'