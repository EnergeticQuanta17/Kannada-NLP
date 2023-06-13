import subprocess

subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', input("Commit Message:")])
subprocess.run(['git', 'push'])
