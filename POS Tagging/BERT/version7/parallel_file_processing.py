import concurrent.futures

def process_line(line):
    pass

def main():
    with open('kn_1k.txt', 'r') as f:
        lines = f.readlines()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for line in lines:
            futures.append(executor.submit(process_line, line))

    for future in futures:
        print(future.result())

if __name__ == '__main__':
    main()
