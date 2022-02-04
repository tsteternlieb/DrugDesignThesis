from rewards import SingleReward, SizeReward

class Profiler():
    def __init__(self,path):
        self.path = path

    def profile(self, rewardModule: SingleReward):
        """running a reward module against molecule database

        Args:
            rewardModule (SingleReward): rewards module to be profiled
        """

        