import unittest
import logging

from percepteur.recorder import Recorder

logging.basicConfig(level=logging.INFO)


class TestRecorder(unittest.TestCase):

    def setUp(self):
        self.recorder = Recorder.factory(title='Dofus')

    def test_stream(self):
        self.recorder.stream()
