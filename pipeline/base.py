from abc import ABC, abstractmethod


class CorePlaceStep(ABC):
    """
    Abstract base class for a data processing pipeline stage.
    Subclasses must implement the `run` method.
    """

    @abstractmethod
    def run(self, pipe_state: dict) -> dict:
        """
        Process the input pipe_state and return the updated state.
        """
        pass

    def __repr__(self) -> str:
        """
        Returns the name of the processing step.
        Subclasses can override this method if needed.
        """
        return self.__class__.__name__
