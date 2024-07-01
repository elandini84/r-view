import sys

from yarp import ResourceFinder
from yarp import Network

from r_view_code.ImageAnalyzer import ImageAnalyzer


if __name__ == "__main__":
    rf = ResourceFinder()
    rf.setVerbose(True)
    rf.configure(sys.argv)
    network = Network()
    network.init()
    analyzer = ImageAnalyzer()
    analyzer.runModule(rf)