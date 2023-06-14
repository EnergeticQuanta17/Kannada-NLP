import subprocess
import winsound

subprocess.run(['git', 'add', '.'])
subprocess.run(['git', 'commit', '-m', input('Enter the commit message: ')])
result = subprocess.run(['git', 'push'])
print(result)

if result.returncode == 0:
    print("Push successful")
    winsound.Beep(500, 10)
else:
    print("Push failed")
    winsound.Beep(500, 100)