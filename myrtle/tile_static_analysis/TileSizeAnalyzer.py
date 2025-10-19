from tile_static_analysis.utils import roundUpToNearestMultipleOf, MatmulInputs, TileSizes, HardwareLoop, unrollAndJamFactor, EnclosingSCFLoop, unrollAndJamOuterLoops
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

