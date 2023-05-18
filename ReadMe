ssh pi@192.168.1.11
ssh pi@192.168.1.10

password for all raspberry pi: pi


data collection instruction:

00. Make sure PC and raspberry pi are conncted to the same

0. raspi FPS: close to 8

1. match time on both pi before start collecting data, so that the name of kinect data and raspi data will match

2. ssh to raspi, edit timestamp_xxx.py and change capture number on raspberry pi to 3000; kinect's FPS was set to 15, so kinect should collect: 3000x2 = 6000 images

3. run collect.py, this will start raspi and kinect

4. repeat experiment until the LED in the soft body turned off (meaning the raspberry pi has finished image taking)

5. run data_processing_ransac_noise_icp.py or data_processing_disk_pcd.py to process data (order of ICP is slightly different)

5.5 data_processing_disk_rgbd.py can process colored pcd, but the result pcd is too sparse.(not ideal) 

--processed point cloud can be saved as .npz or .ply--





--some useful commands--

copy image from raspberry pi:
scp -r pi@192.168.1.11:~/Collection/IMG/ /home/nuc/Ziqin\ Yuan\ DeepSoRo/
scp -r pi@192.168.1.10:~/Collection/IMG/ /home/nuc/Ziqin\ Yuan\ DeepSoRo/

count file number:
ls | wc -l

remove all file:
rm -r *


