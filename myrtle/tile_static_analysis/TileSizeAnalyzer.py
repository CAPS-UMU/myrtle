from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, EnclosingSCFLoop
import pandas as pd
import pathlib
from abc import ABC, abstractmethod


class TileSizeAnalyzer(ABC):

    @abstractmethod
    def analyze_options(self):
        pass

    @abstractmethod
    def analyze_option(self):
        pass

    @abstractmethod
    def exportAnalysisToCSV(self):
        pass

    @abstractmethod
    def unrollAndJamFactor(self):
        pass
        
    @abstractmethod
    def unrollAndJamOuterLoops(self):
        pass

