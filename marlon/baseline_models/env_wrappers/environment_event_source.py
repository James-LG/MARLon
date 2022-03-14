from abc import abstractmethod
from typing import List

class IEnvironmentObserver:
    """
    Interface for classes that want to subscribe to environment events.
    """
    @abstractmethod
    def on_reset(self, last_reward):
        """ Notified when environment is reset. """
        raise NotImplementedError

class EnvironmentEventSource:
    def __init__(self):
        self.observers: List[IEnvironmentObserver] = []

    def add_observer(self, observer: IEnvironmentObserver):
        self.observers.append(observer)

    def notify_reset(self, last_reward):
        # Notify all observers that a reset was called.
        for observer in self.observers:
            observer.on_reset(last_reward)
