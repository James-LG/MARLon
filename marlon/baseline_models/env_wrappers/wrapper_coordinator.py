from abc import abstractmethod
from typing import List, Optional

from cyberbattle import simulation
from cyberbattle._env.cyberbattle_env import CyberBattleEnv, Observation, AttackerGoal, DefenderConstraint, DefenderGoal
from cyberbattle._env.defender import DefenderAgent

class ICyberBattleEnvObserver:
    @abstractmethod
    def on_reset(self):
        raise NotImplementedError

class WrapperCoordinator:
    def __init__(self):
        self.observers: List[ICyberBattleEnvObserver] = []

    def add_observer(self, observer: ICyberBattleEnvObserver):
        self.observers.append(observer)

    def notify_reset(self):
        # Notify all observers that a reset was called.
        for observer in self.observers:
            observer.on_reset()
