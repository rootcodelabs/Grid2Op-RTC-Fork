import numpy as np
from grid2op.Reward.baseReward import BaseReward
from grid2op.dtypes import dt_float
import math


class ShapedReward(BaseReward):
    """
    This reward is based on the cumulative sum of all overflowing line loads, which the agent aims to minimize.
    
    we first calculate the coefficient u, which summarizes the (overflowing) line loads.
    If Rho_max < 1,i.e, there is currently no overflow, and line loads of all lines are within the allowed bounds, u is
    calculated as
    
        u = max(Rho_max-0.5, 0) ( If Rho_max - 0.5 is positive or zero, it will return Rho_max - 0.5, else it will return 0.)
    
    If Rho_max > 1,  u is calculated as
    
        u = sum of (rho_i - 0.5) for each i in the range [1, n] where rho_i > 1, n is the number of power lines in the grid
        
    Then, utilizing u calculated above, we take into account offline lines and apply exponential decay to obtain
    the shaped reward r as
        
        r =exp(-u - 0.5*n_offline)
        
    n_offline is the number of lines which are currently offline as a result of an overflow or agent’s actions (i.e.,
    we do not consider lines that are offline because of maintenance or opponent attacks)
        
    """
    def __init__(self, logger=None):
        BaseReward.__init__(self, logger=logger)

        
    def initialize(self, env):
        self.reward_min = dt_float(0.0)
        
    
    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        
        if not is_done and not has_error:
            res = self.line_overflowing_sum_reward(env)
            
        else:
            res = self.reward_min
            
        return res
    
    
    @staticmethod
    def line_overflowing_sum_reward(env):
        obs = env.current_obs
        
        if obs.rho.max()<=1:
            u = max([obs.rho.max()- 0.5, 0])
            
        else:
            u = sum([rho-0.5 for rho in obs.rho if rho>1])
            
        lines_disconnected = np.sum(obs.line_status == False)
        lines_in_maintenance = np.sum(obs.time_next_maintenance==0)
        lines_under_attack = np.sum(obs.time_since_last_attack>=0)
        n_offline = lines_disconnected - lines_in_maintenance - lines_under_attack
            
        reward = math.exp(-u-0.5*n_offline)
            
        return reward
    