import subprocess

# Command to execute
command = 'dir'

# Run the command and display the output
result = subprocess.run(command, shell=True, capture_output=False)

# Print the output
print(result.stdout.decode())
