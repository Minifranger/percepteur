import logging

logger = logging.getLogger(__name__)


class Factory:
    instance = None

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def factory(cls, **kwargs):
        if cls.instance is None:
            cls.instance = cls(**kwargs)
        assert isinstance(cls.instance, cls)
        return cls.instance
