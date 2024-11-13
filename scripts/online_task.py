from scripts.configs import ConfigMaster

from scripts.online_continuous_task import OnlineCTTraining
from scripts.online_episodic_task import OnlineETTraining


if __name__ == "__main__":
    config_master = ConfigMaster()
    if config_master.config_general.continuous_task:
        OnlineCTTraining(config_master)
    else:
        OnlineETTraining(config_master)