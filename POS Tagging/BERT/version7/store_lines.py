input_file = 'kn.txt'
output_file = 'kn_10k.txt'
line_limit = 10_000

with open(input_file, 'r') as input_f, open(output_file, 'w') as output_f:
    for i, line in enumerate(input_f, start=1):
        if i > line_limit:
            break
        output_f.write(line)
