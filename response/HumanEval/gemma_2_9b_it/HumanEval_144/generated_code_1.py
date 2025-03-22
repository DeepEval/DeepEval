def method():
    def simplify(x, n):
      x_num = int(x.split('/')[0])
      x_den = int(x.split('/')[1])
      n_num = int(n.split('/')[0])
      n_den = int(n.split('/')[1])

      # Calculate the product of numerators and denominators
      product_num = x_num * n_num
      product_den = x_den * n_den

      # Check if the product is a whole number
      return product_num % product_den == 0

    # Test Cases
    print(simplify("1/5", "5/1"))  # Output: True
    print(simplify("1/6", "2/1"))  # Output: False
    print(simplify("7/10", "10/2")) # Output: False
    
    return True 

# Call the function 
output = method()