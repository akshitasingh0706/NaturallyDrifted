from math import log2
import numpy as np
from scipy import stats
import pandas as pd
from typing import Callable, Dict, Optional, Union
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from transformers import AutoTokenizer
from functools import partial
import nlp

from alibi_detect.cd import KSDrift, MMDDrift, LearnedKernelDrift, ClassifierDrift, LSDDDrift
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.models.tensorflow import TransformerEmbedding
from alibi_detect.cd.tensorflow import UAE
from alibi_detect.cd.tensorflow import preprocess_drift

from sampling import samplingData
from base import detectorParent
from basicDetectors import basicDetectors
from onlineDetectors import onlineDetectors
from contextDetectors import contextDetectors

class allDetectors(onlineDetectors, contextDetectors, basicDetectors):
    def run(self):
        if self.test in ["MMD", "LSDD"] and self.drift_type in ['Sudden', 'Gradual']:
            return basicDetectors.run(self)
        elif self.test in ["MMD", "LSDD"] and self.drift_type == "Online":
            return onlineDetectors.run(self)
        elif self.test in ["MMD", "LSDD"] and self.context_type is not None:
            return contextDetectors.run(self)
        else:
            print("Please look into the test or drift type")


