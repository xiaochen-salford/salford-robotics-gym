from srg.agents.policy_gradient_agents.ppo_single import PPOSingle

class PosePPO(PPOSingle):
    agent_name = 'PosePPO'

    def __init__(self, config):
        PPOSingle.__init__(self, config)

    def get_state_size(self):
        return self.config.environment.get_state_size()

    def get_environment_title(self):
        return "Pose Learning"

    def get_lowest_possible_episode_score(self):
        return -100
    
    def get_score_required_to_win(self):
        return -1

    def get_trials(self):
        return 100

   