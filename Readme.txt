scp -r pi@192.168.1.11:~/Collection/IMG/ /home/nuc/Ziqin\ Yuan\ DeepSoRo/
scp -r pi@192.168.1.10:~/Collection/IMG/ /home/nuc/Ziqin\ Yuan\ DeepSoRo/

ssh pi@192.168.1.11
ssh pi@192.168.1.10

password for raspberry pi: pi




data collection instruction:
0. raspi FPS: close to 8
1. match time on both pi before start collecting data, so that the name of kinect data and raspi data will match
2. ssh to raspi, edit timestamp_xxx.py and change capture number on raspberry pi to 3000; with kinect's FPS was set to 15, so kinect should collect: 3000x2 = 6000 images
3. run 00_collect.py, this will start raspi and kinect
4. repeat experiment until the LED in tha softbody turned off
5. run 00_data_processing_ransac_noise_icp.py or 00_data_processing_disk_icp.py to process data

5.5 00_data_processing_disk_rgbd.py can process colored pcd, but the result pcd is too sparse. 



some useful commands

count file number:
ls | wc -l

remove all file:
rm -r *

set time on pi:
sudo date -s '2023-05-05 HH:MM:SS'
