# FLAVORS = [
#     "Banana",
#     "Chocolate",
#     "Lemon",
#     "Pistachio",
#     "Raspberry",
#     "Strawberry",
#     "Vanilla",
# ]

# for i in range(len(FLAVORS)):
#     for j in range(i + 1, len(FLAVORS)):
#         print(f"{FLAVORS[i]}, {FLAVORS[j]}")




#is_prime
# import math
# def is_prime(n):
#     if n <= 1:
#         return False
#     for i in range(2, int(math.sqrt(n)) + 1):
#         if n % i == 0:
#             return False
#     return True

# print(is_prime(1))
# print(is_prime(2))
# print(is_prime(3))
# print(is_prime(4))
# print(is_prime(20021663))  # Test with a large number






#sort students by name
# students = [
#     {"name": "John", "age": 20},
#     {"name": "Jane", "age": 21},
#     {"name": "Jim", "age": 22},
# ]
# #sort students by name
# students.sort(key=lambda x: x["name"])
# print(students)

# students = [(85, "Susan"), (37, "Jeanette"), (6, "Joshua")]
# students.sort(key=lambda x: x[0], reverse=True)
# print(students)



# import math
# prime_list = []
# for i in range(10000, 10050):
#     flag=0
#     for j in range(2, int(math.sqrt(i))+1):
#         if i % j == 0:
#             flag = 1
#     if flag==0:
#         prime_list.append(i)
            
# string_list = " ".join(str(i) for i in prime_list)
# print(string_list)





# import itertools
# import math
# import sys

# def is_prime(n):
#     if n <= 1:
#         return False
#     if n == 2:
#         return True
#     if n % 2 == 0:
#         return False
#     for i in range(3, int(math.sqrt(n)) + 1, 2):
#         if n % i == 0:
#             return False
#     return True

# # Start checking from the first number greater than 100,000,000
# for number in itertools.count(100_000_001):
#     if is_prime(number):
#         print(number)
#         sys.exit()  # Exit after finding the first prime

#reverse Roman numerals
# roman_numerals = {
#     'I': 1,
#     'V': 5,
#     'X': 10,
#     'L': 50,
#     'C': 100,
#     'D': 500,
#     'M': 1000
# }

# def roman_to_integer(s):
#     total = 0
#     prev_value = 0
#     for char in s:
#         value = roman_numerals[char]
#         if value > prev_value:
#             total += value - 2 * prev_value
#         else:
#             total += value
#         prev_value = value
#     return total




#Display the date and time
# Created on Dec. 16, 2016 by Julien Palard
# It should look like this:
# Today is 2015-09-17 and it is 09:34:35

# import time
# from datetime import datetime

# # Get current date and time
# current_datetime = datetime.now()

# # Format the date and time as specified
# date_str = current_datetime.strftime("%Y-%m-%d")
# time_str = current_datetime.strftime("%H:%M:%S")

# # Display the formatted date and time
# print(f"Today is {date_str} and it is {time_str}")






#The missing card

# Write a function named missing_card that given a card game returns the (single) missing card name.
# The card game will be given as a single string of space-separated cards names.
# A card is represented by its color and value, the color being in {"S", "H", "D", "C"} and the value being in {"2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"}, for a total of 52 possibilities.
# You'll always be given 51 cards, and you have to return the missing one.

# def missing_card(cards_string):
#     # Define all possible cards
#     colors = ["S", "H", "D", "C"]
#     values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
    
#     # Create a set of all possible cards
#     all_cards = {color + value for color in colors for value in values}
    
#     # Convert the input string to a set of cards
#     given_cards = set(cards_string.split())
    
#     # Find the missing card (the difference between all possible cards and given cards)
#     missing = all_cards - given_cards
#     # Return the missing card (there should be only one)
#     return missing.pop()    

# # Example usage:
# cards = "S2 S3 S4 S5 S6 S7 S8 S9 S10 SJ SQ SK SA H2 H3 H4 H5 H6 H7 H8 H9 H10 HJ HQ HK HA D2 D3 D4 D5 D6 D7 D8 D9 D10 DJ DQ DK DA C2 C3 C4 C5 C6 C7 C8 C9 C10 CJ CQ CK"
# print(missing_card(cards))  # Output: "CA"






#Friday the 13th
# Find the next friday the 13th.
# Write a function named friday_the_13th, which takes no parameter, and just returns the date of the next friday the 13th.
# If today is a friday the 13th, return it, not the next one.
# Return the date as a string of the following format: YYYY-MM-DD.

# import datetime

# def friday_the_13th():
#     # Get today's date
#     today = datetime.date.today()
    
#     # First check if today is the 13th and a Friday
#     if today.day == 13 and today.weekday() == 4:
#         return today.strftime("%Y-%m-%d")
    
#     # If today is before the 13th of the current month, check this month's 13th
#     if today.day < 13:
#         check_date = datetime.date(today.year, today.month, 13)
#         if check_date.weekday() == 4:  # Friday is weekday 4
#             return check_date.strftime("%Y-%m-%d")
    
#     # Start from the current month
#     month = today.month
#     year = today.year
    
#     # Check each month until we find a Friday the 13th
#     while True:
#         # Move to the next month
#         month += 1
#         if month > 12:
#             month = 1
#             year += 1
        
#         # Create a date for the 13th of the month
#         check_date = datetime.date(year, month, 13)
        
#         # Check if it's a Friday
#         if check_date.weekday() == 4:  # Friday is weekday 4
#             return check_date.strftime("%Y-%m-%d")

# # Example usage:
# print(friday_the_13th())



# import datetime

# print(datetime.date.today())
# print(datetime.date.today().weekday())
# print(datetime.date.today().year)
# print(datetime.date.today().month)
# print(datetime.date.today().day)









