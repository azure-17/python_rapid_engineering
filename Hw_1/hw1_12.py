#Python for Rapid Engineering Solutions
#            Homework #1
#
# Newman 2.12 list of primes
# @author: Isha Joshi.

import math                            # Importing the library for square root

# Function to print primes from 2 to 10,000
# Using the optimized sqaure root method described in Newman 2.12
def print_primes():
    
    prime_number_list = [2]               # Initialize the list with 2 as it is the first prime
    
    for prime_candidate in range(3, 10000):             # Start looping from 3 to 10,000 to test for primes
        is_prime = True                      # Initialize a boolean is_prime to true
    
        for prime_factor in prime_number_list:   # Loop the elements of the prime_number_list (since we're goin in sequence)
            if prime_factor <= int(math.sqrt(prime_candidate)):     # We only need to go until sqrt(prime_candidate)
                if prime_candidate % prime_factor == 0:         # Verify if the prime_candidate has a factor that divdes is completely
                    is_prime = False                      # If yes, then set the flag to false as the candidate is not prime
                    break                                   # Break the loop as we do not need to test more, it is not prime.
            else:
                break                                   # Don't need to test numbers > sqrt(n)
        
        if (is_prime):                               # Outside the inner loop, validate if the flag is still true, if it is, append it to prime list
            prime_number_list.append(prime_candidate)
    
    print(prime_number_list)                  # Finally, print the list of the primes.



# Starting point of the script
print_primes()
