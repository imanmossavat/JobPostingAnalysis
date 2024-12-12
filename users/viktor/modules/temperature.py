from interfaces import ISoftmaxTransformer
import numpy as np
from scipy.special import softmax

class SoftmaxWithTemperature(ISoftmaxTransformer):
    def __init__(self, temp: float):
        self.temp = temp

    def apply(self, row: np.ndarray) -> np.ndarray:
        """
        Apply softmax transformation to the given row using the specified temperature.
        
        Args:
            row: A numpy array representing the row of features to be transformed.
        
        Returns:
            A numpy array after applying softmax with temperature.
        """
        return softmax(row / self.temp)

    def set_temperature(self, temp: float):
        """
        Set the temperature for softmax transformation.
        
        Args:
            temp: A float representing the temperature to be applied.
        """
        self.temp = temp