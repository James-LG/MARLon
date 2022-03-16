from abc import abstractmethod
from typing import List


class IRewardStore():
    """Stores the rewards given by an environment during a single episode."""

    @property
    @abstractmethod
    def episode_rewards(self) -> List[float]:
        """Get a list of rewards for every step in an episode.

        Returns:
            A list of rewards for this episode.
        """
        raise NotImplementedError
