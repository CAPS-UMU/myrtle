from abc import abstractmethod

class TileSizeGenerator:

    @abstractmethod
    def dividesIntoM(self):
        pass

    @abstractmethod
    def dividesIntoN(self):
        pass

    @abstractmethod
    def dividesIntoK(self):
        pass

    @abstractmethod
    def mDimOptions(self):
        pass

    @abstractmethod
    def nDimOptions(self):
        pass

    @abstractmethod
    def kDimOptions(self):
        pass

    @abstractmethod
    def validOptions(self):
        pass

    @abstractmethod
    def computeL1Usage(self):
        pass

    @abstractmethod
    def convertOptionsToDF(self):
        pass

    @abstractmethod
    def exportOptionsToCSV(self):
        pass
