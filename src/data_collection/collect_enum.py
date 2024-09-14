"""Define the enum for data collection."""

from enum import Enum


class CollectEnum(Enum):
    DONE_FALSE = 2  # Data collection in progress.
    SUCCESS = 3  # Successful trajectory is collected.
    FAIL = 4  # Failed trajectory is collected.
    REWARD = 5  # Annotate reward +1.
    SKILL = 6  # Annotate new skill.
    RESET = 7  # Reset environment.
    TERMINATE = 8  # Terminate data collection.
    UNDO = 9  # Undo last 10 actions.
    PAUSE = 10  # Pause data collection.
    CONTINUE = 11  # Continue data collection.
    SUCCESS_RECORD = 12  # Successful trajectory is collected, should save last
