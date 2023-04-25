import paramiko
import time
import os

# Define the IP address and credentials for the Raspberry Pi devices
pi1 = {'ip': '192.168.1.10', 'username': 'pi', 'password': 'pi'}

# Define the programs to run on each Raspberry Pi
program1_0 = 'sudo python3 LED_start.py'
program1_1 = 'python picamera_test.py'
program1_2 = 'sudo python3 LED_end.py'
# Define the directory where the programs are located
program_dir = '/home/pi/Collection/'

# Connect to the Raspberry Pi devices using SSH
ssh1 = paramiko.SSHClient()
ssh1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh1.connect(pi1['ip'], username=pi1['username'], password=pi1['password'])

# Change to the program directory on each Raspberry Pi
stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_0))

# Wait for the programs to finish running
while not stdout1.channel.exit_status_ready():

   time.sleep(1)
   print("turn on LED?", stdout1.channel.exit_status_ready())

stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_1))

while not stdout1.channel.exit_status_ready():

   time.sleep(1)
   print("finish collecting? ", stdout1.channel.exit_status_ready())

stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_2))

while not stdout1.channel.exit_status_ready():

   time.sleep(1)
   print("turn off LED? ", stdout1.channel.exit_status_ready())
# Close the SSH connections
ssh1.close()
# ssh2.close()