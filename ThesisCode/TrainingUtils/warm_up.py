'''Module for warm up lr Scheduler'''
from cgi import test
from os import times
from torch import optim
import matplotlib.pyplot as plt


class WarmUpOptim():
    """Scheduler-esq class which updates adam LR according to a warm up rate
    """
    def __init__(self, optim, start: float, stop: float, steps:int ):
        """Init for warm up scheduler

        Args:
            optim (optim): optim that will be warmed up
            start (float): starting lr
            stop (float): lr to reach after warmup
            steps (int): how many steps to take before stop lr is achevied 
        """
        self.optim = optim
        self._step = 0
        self.start = start
        self.stop = stop 
        self.steps = steps
        self.increment = (self.stop - self.start) / self.steps
        self.logs = []
        
        # for p in self.optimizer.param_groups:
        #     p['lr'] = start
        
    def rate(self):
        rate = self.increment*self._step
        
        return rate
    
    def step(self):
        rate = self.rate()
        print(rate, "RATE")
        if self._step+1 != self.steps:  
            self._step += 1
            for p in self.optim.param_groups:
                p['lr'] = rate
        self.logs.append(rate)
        
        
# test_s = WarmUpOptim(None, 0,3e-4,1000)

# for i in range(2000):
#     test_s.step()
    
# plt.plot(test_s.logs)
# plt.ylabel('some numbers')
# plt.savefig('rates')

        