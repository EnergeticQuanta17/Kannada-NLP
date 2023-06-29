import concurrent.futures
import time

def process_line(line):
    pass

def main():
    with open('kn_1m.txt', 'r') as f:
        lines = f.readlines()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for line in lines:
            executor.submit(process_line, line)

if __name__ == '__main__':
    start = time.perf_counter()
    main()
    print("Done", time.perf_counter() - start)