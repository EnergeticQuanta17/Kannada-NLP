import sys
import subprocess

args = int(sys.argv[1])
print(args)

if(args==1):
    subprocess.run(['python', 'create_indices.py', '--train', 'train.data', '--lang', 'kan'])