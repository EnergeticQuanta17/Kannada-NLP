import subprocess
subprocess.run(['git', 'config', '--global', '--unset', 'http.proxy'])
subprocess.run(['git', 'config', '--global', '--unset', 'https.proxy'])
subprocess.run(['git', 'pull'])
