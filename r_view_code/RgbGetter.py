from yarp import TypedReaderCallbackImageRgb, TypedReaderImageRgb
from yarp import ImageRgb
import threading

class RgbGetter(TypedReaderCallbackImageRgb):
    def __init__(self):
        TypedReaderCallbackImageRgb.__init__(self)
        self.rgb = ImageRgb()
        self.rgb.resize(320, 240)
        self.rgb.zero()
        self.mutex = threading.Lock()

    def onRead(self, image: ImageRgb, reader: TypedReaderImageRgb):
        with self.mutex:
            self.rgb.copy(image)
        return True

    def getImage(self):
        with self.mutex:
            return self.rgb