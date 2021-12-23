import logging
import time
import cv2
import mss
import numpy
from functools import wraps

from percepteur.application import Application
from percepteur.factory import Factory
from workout.image import MSSImage
from workout.labelimg.data import Data
from workout.vision.detection import Detection
from workout.vision.trained_model import TrainedModel

logger = logging.getLogger(__name__)


#DenseToDenseSetOperation : exporter les object detector dans une nouvelle classe detector, ne garder que le recording ici
class Recorder(Factory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Application.factory(**kwargs)
        TrainedModel.factory(**kwargs)
        Detection.factory(**kwargs)
        RecorderStats.factory(**kwargs)

    @property
    def application(self):
        return Application.instance

    @property
    def trained_model(self):
        return TrainedModel.instance

    @property
    def detection(self):
        return Detection.instance

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
    def stream(self, **kwargs):
        image = MSSImage(image=self.grab(**kwargs), application=self.application)
        image = self.detect(model=self.trained_model.model_with_signatures, image=image)
        image = self.draw_boxes(image=image, category_index=Data.instance.category_index)
        cv2.imshow("stream", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    @Decorators.stats(key='grab')
    def grab(self, **kwargs):
        return kwargs.get('sct').grab(self.application.monitor)

    @Decorators.stats(key='detect')
    def detect(self, **kwargs):
        return self.detection.detect(**kwargs)

    @Decorators.stats(key='draw')
    def draw_boxes(self, **kwargs):
        return self.detection.draw_boxes(**kwargs)


class RecorderStats(Factory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        """ keep only the latest length latency """
        self.length = kwargs.get('length', 100)
        self.stats = {'stream': numpy.zeros(self.length), 'grab': numpy.zeros(self.length),
                      'detect': numpy.zeros(self.length), 'draw': numpy.zeros(self.length)}

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
                                                                      round(numpy.mean(self.stats.get('stream'))*1000, 2)), (0, 30), font,
                    fontScale, fontColor, thickness, lineType)
        cv2.putText(blank_image, 'grab latency : %s, mean latency : %s' % (round(self.stats.get('grab')[0]*1000, 2),
                                                                      round(numpy.mean(self.stats.get('grab'))*1000, 2)), (0, 60), font,
                    fontScale, fontColor, thickness, lineType)
        cv2.putText(blank_image, 'detect latency : %s, mean latency : %s' % (round(self.stats.get('detect')[0]*1000, 2),
                                                                      round(numpy.mean(self.stats.get('detect'))*1000, 2)), (0, 90), font,
                    fontScale, fontColor, thickness, lineType)
        cv2.putText(blank_image, 'draw latency : %s, mean latency : %s' % (round(self.stats.get('draw')[0]*1000, 2),
                                                                      round(numpy.mean(self.stats.get('draw'))*1000, 2)), (0, 120), font,
                    fontScale, fontColor, thickness, lineType)
        cv2.imshow("Black Blank", blank_image)
