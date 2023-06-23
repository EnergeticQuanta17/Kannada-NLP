count_words = 0
count_lines = 0
with open('kn.txt', 'r') as f:
    for line in f:
        count_lines+=1

        count_words += len(line.split(' ')) + 1
print(count_lines, count_words)