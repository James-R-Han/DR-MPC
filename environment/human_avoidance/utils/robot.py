# original file: https://github.com/Shuijing725/CrowdNav_Prediction_AttnGraph

from environment.human_avoidance.utils.agent import Agent
from environment.human_avoidance.utils.state import JointState


class Robot(Agent):
    def __init__(self, config,section):
        super().__init__(config,section)
        self.sensor_range = config.robot.sensor_range

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action


    def actWithJointState(self,ob):
        action = self.policy.predict(ob)
        return action
