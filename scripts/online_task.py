from scripts.configs import ConfigMaster

from scripts.online_continuous_task import OnlineCTTraining
from scripts.online_episodic_task import OnlineETTraining


if __name__ == "__main__":
    if config_master.config_general.continuous_task:
        OnlineCTTraining()
    else:
        OnlineETTraining()