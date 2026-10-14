policy_factory = dict()
def none_policy():
    return None

from scripts.policy.orca import ORCA
from scripts.policy.social_force import SOCIAL_FORCE

policy_factory['orca'] = ORCA
policy_factory['none'] = none_policy
policy_factory['social_force'] = SOCIAL_FORCE # From previous codebase, I have not manually verified use of SFM.
