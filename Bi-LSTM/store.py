import sys

with open('../../all_outputs.txt') as f:
    for line in sys.stdin:
        if(line.startswith('2023')):
            continue

        f.write(line)