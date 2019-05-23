
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app
import random
from enhancedbaseagentddqnTstableYun import *
# import enhancedbaseagentYun
from modified_state_space import state_modifier
import numpy as np

class terran_agent(EnhancedBaseAgent):



    def __init__(self):
        super(terran_agent, self).__init__()
        self.attack_coordinates = None
        self.iteration = 0
        
        
        
    def step(self, obs):
    
        simp_obs = state_modifier.modified_state_space(obs)
    
    
        self.iteration += 1
        super(terran_agent, self).step(obs) # crashes with simple obs
        
        if self.unit_type_is_selected(obs, units.Terran.Marine) and self.iteration % 20 == 0:
            return self.handle_action(obs)
            
        else:
            return self.select_units_by_type(obs, units.Terran.Marine)
            
        return actions.FUNCTIONS.no_op()
        
        
        
        
     # Implement this     
    def handle_action(self, 
                        obs):
        x = np.random.randint(0,4)
        y = np.random.randint(0,4)
        if np.random.randint(2)==1:
            return actions.FUNCTIONS.Move_screen("now", (x*21 + 10,y*14 +9 ))
        else:
            return actions.FUNCTIONS.Attack_screen("now", (x*21 + 10,y*14 +9 ))




def main(unused_argv):
    agent = terran_agent()
    try:
        DZaB_data = run_game_with_agent(agent, "DefeatZerglingsAndBanelings", 10)
        # DR_data = enhancedbaseagent.run_game_with_agent(agent, "DefeatRoaches", 10)
                    
    except KeyboardInterrupt:
        pass
        
if __name__ == "__main__":
    app.run(main)


