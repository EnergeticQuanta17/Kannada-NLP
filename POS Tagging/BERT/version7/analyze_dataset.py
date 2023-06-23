from tqdm import tqdm
from collections import Counter
import json

count_words = 0
count_lines = 0

empty_lines = 0

words_set = set()
list_words = []
with open('kn.txt', 'r') as f:
    total_lines = 67_265_080


    for i, line in tqdm(enumerate(f), total=total_lines):
        count_lines+=1

        words = line.strip().split(' ')
        try:
            if(words[-1][-1]=='.'):
                temp1 = words[-1][:-1]
                temp2 = words[-1][-1]

                words[-1] = temp1
                words.append(temp2)
        except:
            if(len(words)!=0 and words[-1]!=''):
                print(words)
                print(words[-1], ord(words[-1]))
                input()

        words_set.update(words)
        list_words.extend(words)

counter = Counter(list_words)
most_common = sorted(counter.items(), key=lambda x:x[1], reverse=True)


print("Count of unique words: ", len(words_set))
print("Most frequent words: ")

# most_common = counter.most_common(20)
# for element, count in most_common:
#     print(element, count)


with open('most_frequent_words.json', 'w') as file:
    json.dump(most_common, file)