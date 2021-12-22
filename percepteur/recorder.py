import logging
import time
import cv2
import mss
import numpy
from functools import wraps

from percepteur.application import Application
from percepteur.factory import Factory

logger = logging.getLogger(__name__)


class Recorder(Factory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Application.factory(**kwargs)
        RecorderStats.factory(**kwargs)

    @property
    def application(self):
        return Application.instance

    @property
    def recorder_stats(self):
        return RecorderStats.instance

    class Decorators:
        @classmethod
        def stats(cls, key=None):
            def decorator(f):
                @wraps(f)
                def wrapper(self, *args, **kwargs):
                    start = time.time()
                    result = f(self, *args, **kwargs)
                    self.recorder_stats.update_stats(key=key, stat=time.time() - start)
                    return result
                return wrapper
            return decorator

        @classmethod
        def record(cls, record_stats=False):
            def decorator(f):
                @wraps(f)
                def wrapper(self, *args, **kwargs):
                    logger.info('Creating mss screenshot instance')
                    with mss.mss() as sct:
                        logger.info('Recording indefinitely')
                        while True:
                            result = f(self, sct=sct, *args, **kwargs)
                            if record_stats:
                                self.recorder_stats.record()
                            if cv2.waitKey(25) & 0xFF == ord("q"):
                                cv2.destroyAllWindows()
                                logger.info('Stop recording')
                                break
                    return result
                return wrapper
            return decorator

    @Decorators.record(record_stats=True)
    @Decorators.stats(key='stream')
    def stream(self, sct=None):
        image = numpy.array(sct.grab(self.application.monitor))
        cv2.imshow("stream", image)
        return image


class RecorderStats(Factory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ keep only the latest l latency """
        self.length = kwargs.get('length', 100)
        self.stats = {'stream': numpy.zeros(self.length)}

    def update_stats(self, **kwargs):
        key, stat = kwargs.get('key'), kwargs.get('stat')
        self.stats[key] = numpy.roll(self.stats[key], 1)
        self.stats[key][0] = stat

    def record(self, **kwargs):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (0, 30)
        fontScale = 1
        fontColor = (255,255,255)
        thickness = 1
        lineType = 2
        # black blank image
        blank_image = numpy.zeros(shape=[256, 1080, 3], dtype=numpy.uint8)
        # print(blank_image.shape)
        cv2.putText(blank_image, 'latency : %s, mean latency : %s' % (round(self.stats.get('stream')[0]*1000, 2),
                                                                      round(numpy.mean(self.stats.get('stream'))*1000, 2)), bottomLeftCornerOfText, font,
                    fontScale, fontColor, thickness, lineType)
        cv2.imshow("Black Blank", blank_image)
