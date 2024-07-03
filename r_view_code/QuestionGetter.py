from yarp import BottleCallback, TypedReaderBottle
from yarp import Bottle
import threading

from .SkelAnalyzer import SkellAnalyzer

class QuestionGetter(BottleCallback):
    def __init__(self, skelAnalyzer: SkellAnalyzer):
        BottleCallback.__init__(self)
        self.bottle = Bottle()
        self.mutex = threading.Lock()
        self.analyzer = skelAnalyzer

    def onRead(self, bottle: Bottle, reader: TypedReaderBottle):
        with self.mutex:
            self.bottle = bottle
            self.analyzer.manageQuestion(bottle.get(0).asString())
        return True

    def getBottle(self):
        with self.mutex:
            return self.bottle