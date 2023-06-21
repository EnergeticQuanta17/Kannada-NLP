import os
import pickle
import re

from dataset_parser import extract_list_of_sentences

file_names = os.listdir(r'.\Dataset\DL-DL MT\DL-DL MT')

prev_fourth_number = None
sum = 0
for file in file_names:
    numbers = re.findall(r'\d+', file)

    first_number = numbers[0]
    second_number = numbers[1]
    third_number = numbers[2]
    fourth_number = numbers[3]
    
    # print(f"First number: {first_number}")
    # print(f"Second number: {second_number}")
    # print(f"Third number: {third_number}")
    # print(f"Fourth number: {fourth_number}")

    sum += int(fourth_number) - int(third_number) + 1
    # print()

    if(prev_fourth_number is None):
        prev_fourth_number = fourth_number
        continue
    

    if(int(third_number) < int(prev_fourth_number)):
        print(third_number, prev_fourth_number)
        print(file)
    prev_fourth_number = fourth_number

print(sum)