from tqdm import tqdm

count_words = 0
count_lines = 0

words_set = set()
with open('kn.txt', 'r') as f:
    total_lines = 67_265_080


    for i, line in tqdm(enumerate(f), total=total_lines):
        count_lines+=1

        words = line.strip().split(' ')
        print(words[-1])
        temp1 = words[-1][:-1]
        temp2 = words[-1][-1]

        words[-1] = temp1
        words.append(temp2)
        print(temp2)

        if(i==10):
            raise Exception
        words_set.update()
print(count_lines, count_words)