import unittest
import logging

from percepteur.recorder import Recorder
from workout.labelimg.data import Data

logging.basicConfig(level=logging.INFO)


class TestRecorder(unittest.TestCase):
    dofus = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus' #TODO : move this percepteur library
    dofus_graph = r'C:\Users\Minifranger\Documents\python_scripts\workout\workout\dofus\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\graph' #TODO : move this percepteur library

    def setUp(self):
        self.data = Data.factory(source=self.dofus)
        # self.trained_model = TrainedModel.factory(source=self.dofus_graph)
        self.recorder = Recorder.factory(title='Dofus', source=self.dofus_graph)

    def test_stream(self):
        self.recorder.stream()
