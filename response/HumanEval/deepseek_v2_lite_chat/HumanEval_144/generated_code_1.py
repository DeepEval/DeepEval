from decimal import Decimal, getcontext

def simplify(x_str, n_str):
    # Set the precision for the Decimal operations
    getcontext().prec = 10  # You can adjust this precision as needed
    
    # Convert string fractions to Decimal
    x_numerator, x_denominator = map(Decimal, x_str.split('/'))
    n_numerator, n_denominator = map(Decimal, n_str.split('/'))
    
    # Calculate the product of the two fractions
    product = x_numerator * n_numerator, x_denominator * n_denominator
    
    # Check if the product is a whole number
    if product[0].to_eng_string() == product[1].to_eng_string():
        return True
    else:
        return False

# Test cases
print(simplify("1/5", "5/1"))  # Should return True
print(simplify("1/6", "2/1"))  # Should return False
print(simplify("7/10", "10/2"))  # Should return False