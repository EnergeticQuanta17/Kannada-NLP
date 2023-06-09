import subprocess

subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', input('Enter the commit message: ')])
subprocess.run(['git', 'push',])