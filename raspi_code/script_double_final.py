import paramiko
import time
import os

# Define the IP address and credentials for the Raspberry Pi devices
pi1 = {'ip': '192.168.1.10', 'username': 'pi', 'password': 'pi'} #192.168.1.10
pi2 = {'ip': '192.168.1.11', 'username': 'pi', 'password': 'pi'}

# Define the programs to run on each Raspberry Pi
program1_0 = 'sudo python3 LED_start.py'
program1_1 = 'python timestamp_picmaera_test.py' # 100 images   timestamp_picmaera_test.py                  python picamera_test.py 
program1_2 = 'sudo python3 LED_end.py'

program2_0 = 'sudo python3 LED_start.py'
program2_1 = 'python timestamp_picmaera_test.py' # 100 images   timestamp_picmaera_test.py                  python picamera_test.py
program2_2 = 'sudo python3 LED_end.py'
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
stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_0))
stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_0))
# Wait for the programs to finish running
while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

   time.sleep(1)
   print("Pi_1 turn on LED?", stdout1.channel.exit_status_ready())
   print("Pi_2 turn on LED?", stdout2.channel.exit_status_ready())

stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_1))
stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_1))

while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

   time.sleep(1)
   print("Pi_1 finish collecting? ", stdout1.channel.exit_status_ready())
   print("Pi_2 finish collecting? ", stdout2.channel.exit_status_ready())

stdin1, stdout1, stderr1 = ssh1.exec_command('cd {}; {}'.format(program_dir, program1_2))
stdin2, stdout2, stderr2 = ssh2.exec_command('cd {}; {}'.format(program_dir, program2_2))

while not stdout1.channel.exit_status_ready() and not stdout2.channel.exit_status_ready():

   time.sleep(1)
   print("Pi_1 turn off LED? ", stdout1.channel.exit_status_ready())
   print("Pi_2 turn off LED? ", stdout2.channel.exit_status_ready())
# Close the SSH connections
ssh1.close()
ssh2.close()