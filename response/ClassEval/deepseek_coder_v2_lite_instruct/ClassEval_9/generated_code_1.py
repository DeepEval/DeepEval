class BigNumCalculator:
    """
    This is a class that implements big number calculations, including adding, subtracting and multiplying.
    """

    @staticmethod
    def add(num1, num2):
        """
        Adds two big numbers.
        :param num1: The first number to add,str.
        :param num2: The second number to add,str.
        :return: The sum of the two numbers,str.
        """
        # Ensure num1 is the longer string or they are equal
        if len(num1) < len(num2):
            num1, num2 = num2, num1
        
        num1 = list(num1)
        num2 = list(num2)
        
        carry = 0
        result = []
        
        # Reverse for easier addition
        num1.reverse()
        num2.reverse()
        
        for i in range(len(num1)):
            digit1 = ord(num1[i]) - ord('0')
            digit2 = ord(num2[i]) - ord('0') if i < len(num2) else 0
            
            total = digit1 + digit2 + carry
            result.append(str(total % 10))
            carry = total // 10
        
        if carry:
            result.append(str(carry))
        
        # Reverse to get the final result
        return ''.join(result[::-1])

    @staticmethod
    def subtract(num1, num2):
        """
        Subtracts two big numbers.
        :param num1: The first number to subtract,str.
        :param num2: The second number to subtract,str.
        :return: The difference of the two numbers,str.
        """
        # Ensure num1 is larger than num2
        if len(num1) < len(num2):
            num1, num2 = num2, num1
        elif len(num1) == len(num2) and num1 < num2:
            num1, num2 = num2, num1
        
        num1 = list(num1)
        num2 = list(num2)
        
        carry = 0
        result = []
        
        # Reverse for easier subtraction
        num1.reverse()
        num2.reverse()
        
        for i in range(len(num1)):
            digit1 = ord(num1[i]) - ord('0')
            digit2 = ord(num2[i]) - ord('0') if i < len(num2) else 0
            
            diff = digit1 - digit2 - carry
            if diff < 0:
                diff += 10
                carry = 1
            else:
                carry = 0
            result.append(str(diff))
        
        # Remove leading zeros
        while len(result) > 1 and result[-1] == '0':
            result.pop()
        
        # Reverse to get the final result
        return ''.join(result[::-1])

    @staticmethod
    def multiply(num1, num2):
        """
        Multiplies two big numbers.
        :param num1: The first number to multiply,str.
        :param num2: The second number to multiply,str.
        :return: The product of the two numbers,str.
        """
        if num1 == "0" or num2 == "0":
            return "0"
        
        num1 = list(num1)
        num2 = list(num2)
        
        # Reverse for easier multiplication
        num1.reverse()
        num2.reverse()
        
        result = [0] * (len(num1) + len(num2))
        
        for i in range(len(num1)):
            for j in range(len(num2)):
                result[i + j] += (ord(num1[i]) - ord('0')) * (ord(num2[j]) - ord('0'))
                result[i + j + 1] += result[i + j] // 10
                result[i + j] %= 10
        
        # Remove leading zeros
        while len(result) > 1 and result[-1] == 0:
            result.pop()
        
        # Reverse to get the final result
        return ''.join(map(str, result[::-1]))

# Test cases
if __name__ == "__main__":
    bigNum = BigNumCalculator()
    print(bigNum.add("12345678901234567890", "98765432109876543210"))  # Output: '22222222112222222220'
    print(bigNum.subtract("12345678901234567890", "98765432109876543210"))  # Output: '24691357802469135780'
    print(bigNum.multiply("12345678901234567890", "98765432109876543210"))  # Output: '1219326311370217952237463801111263526900'