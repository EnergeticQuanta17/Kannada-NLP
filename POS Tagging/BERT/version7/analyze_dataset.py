from tqdm import tqdm

count_words = 0
count_lines = 0
with open('kn.txt', 'r') as f:
    total_lines = sum(1 for _ in f)
    f.seek(0)


    for i, line in tqdm(enumerate(f), total=total_lines):
        count_lines+=1

        count_words += len(line.split(' ')) + 1
print(count_lines, count_words)