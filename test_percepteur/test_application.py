import unittest
from pyrect import Size

from percepteur.application import Application


class TestApplication(unittest.TestCase):

    def setUp(self):
        self.application = Application.factory(title='Dofus')

    def test_size(self):
        assert isinstance(Application.instance.size, Size)
        assert Application.instance.size == (1920, 1080)

    def test_fps(self):
        import time

        import cv2
        import mss
        import numpy

        with mss.mss() as sct:
            # Part of the screen to capture
            # monitor = {"top": 40, "left": 0, "width": 800, "height": 640}
            i = 0
            m = 0

            while "Screen capturing":
                last_time = time.time()

                # Get raw pixels from the screen, save it to a Numpy array
                img = numpy.array(sct.grab(Application.instance.monitor))

                # Display the picture
                cv2.imshow("OpenCV/Numpy normal", img)

                # Display the picture in grayscale
                # cv2.imshow('OpenCV/Numpy grayscale',
                #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
                i += 1
                m += (time.time() - last_time)*1000
                print("monitor: {}".format(Application.instance.monitor))
                # print("lag: {}".format((time.time() - last_time)*1000))
                # print("fps: {}".format(1 / (time.time() - last_time)))
