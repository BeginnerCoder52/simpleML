# import os
# import sys
# from pathlib import Path
# from typing import Any, Dict
# from datetime import dateTime


# def add(a, b):
#     return a + b


# def subtract(a, b):
#     return a - b


# def multiply(a, b):
#     return a * b


# def divide(a, b):
#     if b == 0:
#         return None
#         return "Cannot divide"
#     return a / b


# a = 10
# b = 5
# print(add(a, b))
# print(subtract(a, b))
# print(multiply(a, b))
# print(divide(a, b))

# dummy_data = {
#     "key1": "value1",
#     "key2": "value2",
#     "key3": "value3",
#     "key4": "value4",
#     "key5": "value5",
# }

import pandas as pd
# creating a dataframe
data = {
    "Name": ["John", "Anna", "Peter", "Linda"],
    "Location": ["New York", "Paris", "Berlin", "London"],
    "Age": [24, 13, 53, 33]
}
df = pd.DataFrame(data)
