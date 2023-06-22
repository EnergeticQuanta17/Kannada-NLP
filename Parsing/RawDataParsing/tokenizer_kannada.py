##############################################################
# FOR NOW SPLITTING ONLY ACCORDING TO " "
# Not using this file for BERT-v4
##############################################################


import os
import re
import pickle

unique_characters = []

unremoved_sentences = []

directory_path = r"..\..\Dataset\Raw Data"
for root, directories, files in os.walk(directory_path):
    for filename in files:
        file_path = os.path.join(root, filename)
    
        with open(file_path, 'r', encoding='utf-8') as f:
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
                
                unremoved_sentences.append(words)
                    
                index += 1

unique_characters.sort()
# print(unique_characters)

with open('unique.txt', 'w', encoding='utf-8') as f:
    for u in unique_characters:
        f.write(f"{ord(u)}--> "+ u + '\n')

special_characters = []

with open('special_characters.txt', 'w', encoding='utf-8') as f:
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
        
print("Special Characters:", special_characters)



# Need to take care how the special characters split
regex = "".join(special_characters)
pattern = "[" + re.escape(regex) + "]"
pattern_with_symbols = "[^" + re.escape(regex) + "]+" + "|" + pattern
print("Pattern with Symbols:", pattern_with_symbols)

delimited_sentences = []
with open('out.txt', 'w', encoding='utf-8') as f:
    count = 0
    for sentence in unremoved_sentences:
        temp = []
        for word in sentence:
            try:
                split = re.findall(pattern_with_symbols, word)
                
                temp.extend(split)
                
            except:
                print("GAVE EXCEPTION FOR THIS -->", word)
        
        delimited_sentences.append(temp)
    
print(len(delimited_sentences))

with open('delimited_sentences.pickle', 'wb') as file:
    pickle.dump(delimited_sentences, file)