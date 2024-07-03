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

class LogStyle:
    HEADER = '\033[95m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    COLORS = {"WARNING":'\033[93m',
              "ERROR":'\033[91m',
              "COMPONENT":'\033[96m',
              "DEBUG":'\033[92m',
              "INFO":'\033[94m'}


class ImageAnalyzer(SkellAnalyzer):
    def __init__(self):
        super().__init__()
        self.imagePortName = "/image-analyzer/img:i"
        self.questionPortName = "/image-analyzer/question:i"
        self.answerPortName = "/image-analyzer/answer:o"
        self.prompt = {"role": "system","content":"You are the robot R1. You have a camera over your head and you can describe what your camera is seeing to the user"}
        self.rgb = None
        self.imageSizes = {"width": 640, "height": 480}

    
    def logMe(self, level:str, message:str):
        styled = LogStyle()
        comp = "COMPONENT"
        if level.upper() in styled.COLORS.keys():
            print(f"[{styled.BOLD}{styled.COLORS[level.upper()]}{level.upper()}{styled.ENDC}] " + f"|{styled.BOLD}{styled.COLORS[comp]}r_view_code.ImageAnalizer{styled.ENDC}| " + message)
        else:
            print("[{0}] [{1}] {2}".format(level.upper(),"r_view_code.ImageAnalizer",message))


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

        if rf.check("image-width"):
            self.imageSizes["width"] = rf.find("image-width").asInt32()
        if rf.check("image-height"):
            self.imageSizes["height"] = rf.find("image-height").asInt32()

        self.questionPort = BufferedPortBottle()
        self.answerPort = BufferedPortBottle()
        if not self.imagePort.open(self.imagePortName):
            self.logMe("error", "Failed to open port")
            return False
        # self.imagePort.useCallback(self.rgbGetter)
        self.prepareInnerImage()
        if not self.questionPort.open(self.questionPortName):
            self.logMe("error", "Failed to open port")
            return False
        self.questionPort.useCallback(self.questionGetter)
        if not self.answerPort.open(self.answerPortName):
            self.logMe("error", "Failed to open port")
            return False
        if not rf.check("model"):
            self.logMe("error", "model not found")
            return False
        self.model = rf.find("model").asString()
        if rf.check("prompt_context"):
            prompt_context = rf.find("prompt_context").asString()
        if rf.check("prompt_file"):
            prompt_file = rf.find("prompt_file").asString()
        self.logMe("debug","before prompt")
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
                self.logMe("error", "Prompt file not found")
                return False
            except Exception as e:
                self.logMe("error", str(e))
                return False
        ollama.chat(
            model = self.model,
            messages = [self.prompt]
        )

        self.logMe("debug","Configured")

        return True


    def prepareInnerImage(self):
        self.in_buf_human_array = np.ones((self.imageSizes["height"], self.imageSizes["width"], 3), dtype=np.uint8)
        self.in_buf_human_image = ImageRgb()
        self.in_buf_human_image.resize(self.imageSizes["width"], self.imageSizes["height"])
        self.in_buf_human_image.setExternal(self.in_buf_human_array.data, self.in_buf_human_array.shape[1], self.in_buf_human_array.shape[0])
        self.rgb = np.ones((self.imageSizes["height"], self.imageSizes["width"], 3), dtype=np.uint8)


    def manageQuestion(self, question):
        self.logMe("info","Got question: {0}".format(question))
        image = self.imagePort.read()
        if image is None:
            plainResp = ollama.chat(
                model = self.model,
                messages = [self.prompt,{
                "role": "user",
                "content": question
            }])
            out = self.answerPort.prepare()
            out.addString(plainResp["message"]["content"])
            self.answerPort.write()
            return
        if image.width() != self.imageSizes["width"] or image.height() != self.imageSizes["height"]:
            self.imageSizes["width"] = image.width()
            self.imageSizes["height"] = image.height()
            self.prepareInnerImage()
        self.in_buf_human_image.copy(image)
        human_image = np.copy(self.in_buf_human_array)
        self.rgb = np.copy(human_image)
        _, im_arr = cv2.imencode('.jpg', self.rgb)
        im_bytes = im_arr.tobytes()
        rgb_enc = base64.b64encode(im_bytes)
        seen_response = ollama.chat(
            model = self.model,
            messages = [self.prompt,{
            "role": "user",
            "content": question,
            "images": [rgb_enc] #base64 encoded
            }])
        self.logMe("info","Output "+seen_response["message"]["content"])
        if seen_response is not None:
            seen_text = seen_response["message"]["content"]
            out = self.answerPort.prepare()
            out.clear()
            out.addString(seen_text)
            self.answerPort.write()


    def updateModule(self):
        return True


    def close(self):
        self.imagePort.close()
        self.questionPort.close()
        self.answerPort.close()
        return True
