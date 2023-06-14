import subprocess
import winsound

subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', input('Enter the commit message: ')])
ret = subprocess.run(['git', 'push'])
print(ret)

if ret.returncode == 0:
    print("Push successful")
else:
    print("Push failed")