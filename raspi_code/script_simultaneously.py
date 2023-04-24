import paramiko
import time
import os

# Define the IP address and credentials for the Raspberry Pi devices
pi1 = {'ip': '192.168.1.10', 'username': 'pi', 'password': 'pi'}
pi2 = {'ip': '192.168.1.101', 'username': 'pi', 'password': 'pi'}

# Define the programs to run on each Raspberry Pi
program1 = 'python picamera_test.py'
program2 = 'python picamera_test.py'

# Define the directory where the programs are located
program_dir = '/home/pi/Collection/'

# Connect to the Raspberry Pi devices using SSH
ssh1 = paramiko.SSHClient()
ssh1.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh1.connect(pi1['ip'], username=pi1['username'], password=pi1['password'])

ssh2 = paramiko.SSHClient()
ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh2.connect(pi2['ip'], username=pi2['username'], password=pi2['password'])

# Change to the program directory on each Raspberry Pi
stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1))
stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2))

# Wait for the programs to finish running
while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

   time.sleep(1)
   print("finish running? ", stdout1.channel.exit_status_ready())

# Close the SSH connections

# ssh1.close()
# ssh2.close()