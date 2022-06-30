#Python for Rapid Engineering Solutions
#            Homework #1
#
# Printing fibonacci and factorial
# @author: Isha Joshi.



# calculate factorial of a number n
# factorial of n = n-1 * n-2 * n-3 and so on until 1.
def factorial(n) :
    if n < 1 :                                      # Validate the number is >= 1
        print("Invalid input, should be >= 1")      # Print if number is < 1
        return
    if n == 1:                                      # Base condition, if n = 1 return 1 since factorial of 1 is 1
        return 1
    else :                                          # Otherwise, recursively call factorial for n-1 and multiply with n
        return n * factorial(n-1)                   # This would branch out like n * n-1 * factorial(n-2) because factorial(n-1) = n-1 * factorial(n-2) and so on


# calculates the nth fibonacci number
# fibonacci of n = fibonacci(n-1) + fibonacci(n-2) + so on until 1.

def fibonacci(n) :
    if (n < 0) or (not isinstance(n, int)):                   #Negative numbers or non integers are not allowed
        print("Invalid input, It should be >= 0")
        return
    if n == 0:
        return 0            #Base conditions, fibonacci(0) = 0
    elif n == 1:
        return 1              #Base conditions, fibonacci(1) = 1
    else :
        return fibonacci(n-1) + fibonacci(n-2)     # recursively call fibonacci for n-1 and n-2 (therfore the n would be >= 2 here)

# function to take input, call functions and print output
def begin():
    x = int(input("Enter an integer for a factorial computation: "))     # User input - for factorial computation
    y = int(input("Enter an integer for the Fibonacci number computation: "))     # User input - for fibonacci computation
    
    factorial_string = str(factorial(x))                 # calling the factorial function and converting to string for print
    fibonacci_string = str(fibonacci(y))                     # calling the fibonacci function and converting to string for print
    
    print("")
    
    if (factorial_string != 'None'):     # No need to print in case of bad input
        print("factorial of "+str(x)+" is "+factorial_string)
    if (fibonacci_string != 'None'):     # No need to print in case of bad input
        print("Fibonacci number "+str(y)+" is "+fibonacci_string)
    
    
        


# Starting point of the program
begin()
