import logging
from pyautogui import getWindowsWithTitle

from percepteur.factory import Factory

logger = logging.getLogger(__name__)


class Application(Factory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.title = kwargs.get('title')
        self.window = self._window()
        # self.maximize()

    def _window(self):
        window = getWindowsWithTitle(self.title)
        logger.info('Found {count} window for {title}'.format(count=len(window), title=self.title))
        return next(iter(window), None)

    @property
    def size(self):
        return self.window.size

    @property
    def topleft(self):
        return self.window.topleft

    @property
    def bottomright(self):
        return self.window.bottomright

    @property
    def monitor(self):
        return {"top": self.topleft.y, "left": self.topleft.x, "width": self.size.width, "height": self.size.height}

    def maximize(self):
        self.window.maximize()
