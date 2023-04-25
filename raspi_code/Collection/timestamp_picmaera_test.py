import os, sys
import time
from time import sleep
from picamera import PiCamera
import numpy as np
from datetime import datetime

class PI_CAMERA():
   
   def __init__(self):
      self.camera = PiCamera()
      self.config_cam()

   def config_cam(self):
      self.camera.resolution = (1000, 1000)
      self.camera.framerate = 60
      self.camera.iso = 800
      # wait for automatic gain control to settle
      sleep(2)
      self.camera.shutter_speed = 100000
      self.camera.exposure_mode = 'off'
      awb_gains = self.camera.awb_gains
      self.camera.awb_mode = 'off'
      self.camera.awb_gains = awb_gains
      

   def capture_single(self):
      self.camera.capture('single_preview_new.jpg')

   # def capture_sequence(self, n):
   #    # ~ 4 FPS for capture sequecne without video port
   #    # ~ 25 FPS for capture sequence with video port (30 FPS)
   #    # ~ 44 FPS for capture sequence with video port (60 FPS)
   #     self.camera.capture_sequence(['./IMG/image%02d.jpg' % i for i in range(n)], use_video_port=True)

   def capture_sequence(self, n):
      for i in range(n):
         npdt = np.datetime64(datetime.now())
         dt = npdt.astype(datetime)

         timestamp = dt.strftime('%d_%H%M%S_%f')[:-3]
         filename = "./IMG/image_{}_{}.jpg".format(str(i), timestamp)
         self.camera.capture(filename, use_video_port=True)

   def test_capture_speed(self, n):
      start_time = time.time()
      self.camera.capture_sequence(n)
      print("--- %s FPS ---" % (n/(time.time() - start_time)))

if __name__ == '__main__':

   os.chdir(sys.path[0])

   cam = PI_CAMERA()
   

   # test capture speed
   cam.capture_sequence(50)
   # cam.test_capture_speed(100)

   # test with single image
#   cam.capture_single()

