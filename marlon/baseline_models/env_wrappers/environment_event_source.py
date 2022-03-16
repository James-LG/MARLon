from abc import abstractmethod
from typing import List

class IEnvironmentObserver:
    """Interface for classes that want to subscribe to environment events."""

    @abstractmethod
    def on_reset(self, last_reward: int):
        """Called when environment is reset.
        
        Args:
            last_reward (int): The last recorded reward value of the environment that caused this event.
        """
        raise NotImplementedError

class EnvironmentEventSource:
    """ Source of environment events that observers can attach themselves to. """

    def __init__(self):
        self.observers: List[IEnvironmentObserver] = []

    def add_observer(self, observer: IEnvironmentObserver):
        """Add an observer to be notified on events.

        Args:
            observer (IEnvironmentObserver): The observer subscribing to events from this source.
        """
        self.observers.append(observer)

    def notify_reset(self, last_reward: int):
        """Notify all observers that a reset has occurred.

        Args:
            last_reward (int): The last recorded reward value of the environment that caused this event.
        """
        # Notify all observers that a reset was called.
        for observer in self.observers:
            observer.on_reset(last_reward)
