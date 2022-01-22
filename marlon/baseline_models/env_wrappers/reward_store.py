from abc import abstractmethod
from typing import List


class IRewardStore():
    """ A simple way of accessing the rewards of an AttackerWrapper. """

    @abstractmethod
    def get_episode_rewards(self) -> List[float]:
        """
        Get a list of rewards for every step in an episode.
        Should be reset when new episodes are started.
        """
        raise NotImplementedError
