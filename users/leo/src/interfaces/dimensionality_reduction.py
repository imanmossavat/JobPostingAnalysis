from abc import ABC, abstractmethod


class DimensionalityReduction(ABC):
    """Abstract base class for dimensionality reduction implementations.

    Any class that performs dimensionality reduction must implement this
    interface, specifically the reduce_dimensions, and generate_2d_plot methods.
    """

    @abstractmethod
    def reduce_dimensions(self):
        """Reduce the dimensions of input data.

        This method must be implemented by concrete classes to define
        how the dimensions of the input data are reduced.
        """
        pass

    @abstractmethod
    def generate_2d_plot(self):
        """Generate a 2D plot of the input data.

        This method must be implemented by concrete classes to define
        how a 2D plot of the input data is generated.
        """
        pass
