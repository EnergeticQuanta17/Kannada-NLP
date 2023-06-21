##############################################################
# FOR NOW SPLITTING ONLY ACCORDING TO " "
# Not using this file for BERT-v4
##############################################################


import os
import re

unique_characters = []

directory_path = r"..\..\Dataset\Raw Data"
for root, directories, files in os.walk(directory_path):
    for filename in files:
        file_path = os.path.join(root, filename)
    
        with open(file_path, 'r') as f:
            index = 0
            for line in f:
            
                line = line.strip()

                # Line without symbols
                # symbols_to_remove = ['"', '`']
                # modified_string = line.replace('"', "")
                # for symbol in symbols_to_remove:
                #     modified_string = modified_string.replace(symbol, '')
                
                words = line.split(' ')

                for word_index, word in enumerate(words):
                    for char in word:
                        if(char not in unique_characters):
                            unique_characters.append(char)
                            print(char, f"Sentence Number: {index}", f"Word index: {word_index}")
                    
                index += 1

unique_characters.sort()
print(unique_characters)

with open('unique.txt', 'w') as f:
    for u in unique_characters:
        f.write(f"{ord(u)}--> "+ u + '\n')

special_characters = []

with open('special_characters.txt', 'w') as f:
    for u in unique_characters:
        unicode = ord(u)
        if(unicode>=3202 and unicode<=3311):
            #Kannada letter
            continue
        
        if(('a' <= u <= 'z') or ('A' <= u <= 'Z')):
            #English letter
            continue
            
        if('0' <= u <= '9'):
            # Number
            continue
        
        special_characters.append(u)
        f.write(u + '\n')

# Need to take care how the special characters split

with open('out.txt', 'w') as o:
    directory_path = r"..\..\Dataset\Raw Data"
    for root, directories, files in os.walk(directory_path):
        for filename in files:
            file_path = os.path.join(root, filename)
        
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    for char in line:
                        
                        if(char in special_characters):
                            print(f"This is the special character: {char}")
                            raise Exception
                            regex = "".join(special_characters)
                            pattern = "[ " + re.escape(regex) + "]"
                            result = re.split(pattern, line)

                            print(result)
                            print()
                            print(line)
                            o.write(line + '\n')
                            o.write(file_path)

                            raise Exception