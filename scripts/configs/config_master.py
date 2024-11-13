from .config_general import ConfigGeneral
from scripts.configs.config_PT import ConfigPT
from scripts.configs.config_HA import ConfigHA
from scripts.configs.config_SAC import ConfigSAC
from scripts.configs.config_training import ConfigTraining


class ConfigMaster(object):
    def __init__(self):
        self.config_general = ConfigGeneral()
        self.config_HA = ConfigHA(self.config_general)
        self.config_SAC = ConfigSAC(self.config_general)
        self.config_PT = ConfigPT(self.config_general)
        self.config_training = ConfigTraining(self.config_general)

        self.identifier = self.config_general.model.policy
        self.identifier += f"_{self.config_general.model.size}"
        self.identifier += f"_{'vis' if self.config_HA.robot.visible else 'invis'}"
        self.identifier += f'_LB{self.config_general.env.lookback}'
        self.identifier += f'_OOD{self.config_general.OOD.use_OOD}'
