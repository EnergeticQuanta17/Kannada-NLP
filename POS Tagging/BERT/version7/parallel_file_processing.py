import concurrent.futures

def process_line(line):
    pass

def main():
    with open('kn_1k.txt', 'r') as f:
        lines = f.readlines()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for line in lines:
            executor.submit(process_line, line)

if __name__ == '__main__':
    main()
    print("Done")