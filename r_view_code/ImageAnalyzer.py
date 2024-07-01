from typing import Any
import ollama
import cv2
import base64
import numpy as np

from yarp import Log
from yarp import BufferedPortImageRgb
from yarp import BufferedPortBottle
from yarp import ResourceFinder

from .RgbGetter import RgbGetter, ImageRgb
from .SkelAnalyzer import SkellAnalyzer
from .QuestionGetter import QuestionGetter

class ImageAnalyzer(SkellAnalyzer):
    def __init__(self):
        super().__init__()
        self.imagePortName = "/image-analyzer/img:i"
        self.questionPortName = "/image-analyzer/question:i"
        self.answerPortName = "/image-analyzer/answer:o"
        self.prompt = None
        self.rgb = None

    def configure(self, rf: Any):
        prompt_context = None
        prompt_file = None
        self.rgbGetter = RgbGetter()
        self.questionGetter = QuestionGetter(self)
        if rf.check("port-prefix"):
            self.imagePortName = "/{0}/img:i".format(rf.find("port-prefix").asString())
            self.questionPortName = "/{0}/question:i".format(rf.find("port-prefix").asString())
            self.answerPortName = "/{0}/answer:o".format(rf.find("port-prefix").asString())
        self.imagePort = BufferedPortImageRgb()

        self.in_buf_human_array = np.ones((480, 640, 3), dtype=np.uint8)
        self.in_buf_human_image = ImageRgb()
        self.in_buf_human_image.resize(640, 480)
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1], self.in_buf_human_array.shape[0])
        self.rgb = np.ones((480, 640, 3), dtype=np.uint8)

        self.questionPort = BufferedPortBottle()
        self.answerPort = BufferedPortBottle()
        if not self.imagePort.open(self.imagePortName):
            Log.error("ImageAnalyzer", "Failed to open port")
            return False
        # self.imagePort.useCallback(self.rgbGetter)
        if not self.questionPort.open(self.questionPortName):
            Log.error("ImageAnalyzer", "Failed to open port")
            return False
        self.questionPort.useCallback(self.questionGetter)
        if not self.answerPort.open(self.answerPortName):
            Log.error("ImageAnalyzer", "Failed to open port")
            return False
        if not rf.check("model"):
            Log.error("ImageAnalyzer", "model not found")
            return False
        self.model = rf.find("model").asString()
        if rf.check("prompt_context"):
            prompt_context = rf.find("prompt_context").asString()
        if rf.check("prompt_file"):
            prompt_file = rf.find("prompt_file").asString()
        print("before prompt")
        if prompt_file is not None:
            promptFinder = ResourceFinder()
            if prompt_context is not None:
                promptFinder.setDefaultContext(prompt_context)
            promptPath = promptFinder.findFileByName(prompt_file)
            try:
                with open(promptPath, 'r') as file:
                    promptText = file.read()
                    self.prompt = {"role": "system","content": promptText}
            except FileNotFoundError:
                Log.error("ImageAnalyzer", "Prompt file not found")
                return False
            except Exception as e:
                Log.error("ImageAnalyzer", str(e))
                return False
        ollama.chat(
            model = self.model,
            messages = [self.prompt]
        )

        print("Configured")

        return True

    def manageQuestion(self, question):
        print("\n\nthe questioooon: {0}\n\n".format(question))
        image = self.imagePort.read()
        self.in_buf_human_image.copy(image)
        human_image = np.copy(self.in_buf_human_array)
        self.rgb = np.copy(human_image)
        # self.rgb = cv2.imread("/mnt/c/ExchangeData/place_red_36dp.png")
        print("image read {0}".format(type(self.rgb)))
        _, im_arr = cv2.imencode('.jpg', self.rgb)
        print("image encoded")
        im_bytes = im_arr.tobytes()
        print("image tobytes")
        rgb_enc = base64.b64encode(im_bytes)
        print("\n\nthe rgb: {0}\n\n".format(type(rgb_enc)))
        seen_response = ollama.chat(
            model = self.model,
            messages = [self.prompt,{
            "role": "user",
            "content": question,
            "images": [rgb_enc] #base64 encoded
            }])
        print("Misticanza "+seen_response["message"]["content"])
        if seen_response is not None:
            seen_text = seen_response["message"]["content"]
            out = self.answerPort.prepare()
            out.addString(seen_text)
            self.answerPort.write()

    def updateModule(self):
        return True

    def close(self):
        self.imagePort.close()
        self.questionPort.close()
        self.answerPort.close()
        return True
