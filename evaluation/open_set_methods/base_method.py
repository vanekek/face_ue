from abc import ABC, abstractmethod
import numpy as np


class OpenSetMethod(ABC):
    @abstractmethod
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def setup(self, similarity_matrix: np.ndarray):
        """
        performs anxiliary operations
        """

    @abstractmethod
    def predict(self):
        """
        perfomes predicton
        """

    @abstractmethod
    def predict_uncertainty(self, data_uncertainty: np.ndarray):
        """
        perfomes uncertainty predicton
        """
