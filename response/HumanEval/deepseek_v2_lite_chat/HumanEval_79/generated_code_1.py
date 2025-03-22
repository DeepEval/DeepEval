def method():
    def decimal_to_binary(n):
        return bin(n).replace("0b", "").zfill(8)  # 8 bits are used to represent the decimal number

    def format_binary(bin_str):
        return 'db' + bin_str + 'db'  # Prepend and append 'db'

    # Test case to validate the function
    def test_function():
        # assert format_binary(decimal_to_binary(10)) == 'db00001010db'
        # assert format_binary(decimal_to_binary(255)) == 'db11111111db'
        # assert format_binary(decimal_to_binary(0)) == 'db00000000db'
        # assert format_binary(decimal_to_binary(-1)) == 'db10000000db'  # Note: binary representation for negative numbers
        print(format_binary(decimal_to_binary(10)))

    # Call the test case function
    test_function()

    return format_binary(decimal_to_binary(10))  # Return the final output