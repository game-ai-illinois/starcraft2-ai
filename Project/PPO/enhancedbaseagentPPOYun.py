
"""
Written by Michael McGuire to provide an enhanced base agent with simple repetitive functions. (09 / 29 / 2018)
"""

import pysc2
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import app
import random
from PPO import *
from modified_state_space import state_modifier
import numpy as np
import tensorflow as tf

class EnhancedBaseAgent(base_agent.BaseAgent):

    def __init__(self):
        super(EnhancedBaseAgent, self).__init__()
    
    
    
    """
        Returns True if obs shows that there is a unit of type unit_type currently selected
    """
    def unit_type_is_selected(self, obs, unit_type):
        
        if (len(obs.observation.single_select) > 0 and
          obs.observation.single_select[0].unit_type == unit_type):
            return True
    
        if (len(obs.observation.multi_select) > 0 and
          obs.observation.multi_select[0].unit_type == unit_type):
            return True
    
        return False
        
    

def random_step(timestep_obs):
    action =actions.FUNCTIONS.no_op()
    arg=None #default value for debugging
    arg = np.random.randint(0,32+40+1)
    action_array = trans_action(arg) 
        # x,y =(0,0)
    if arg < 16:
        # move
        y = int(arg/4)
        x= arg %4
        if (  actions.FUNCTIONS.Move_screen.id  in timestep_obs.observation.available_actions ):
            action= actions.FUNCTIONS.Move_screen("now",(x*21 + 10,y*14 +9 ))
    elif arg<32:
        # attack
        y = int((arg%16)/4)
        x= arg %4
        if (  actions.FUNCTIONS.Attack_screen.id  in timestep_obs.observation.available_actions ):
            action = actions.FUNCTIONS.Attack_screen("now", (x*21 + 10,y*14 +9 ))
    elif arg<32+40:
        idx= arg-32
        x = idx%8 #horizontal box number
        y =int(idx/8) #vertical box number
        action = actions.FUNCTIONS.select_rect("select", (x*10, (y+1)*11) , ((x+1)*10, (y)*11) )
    else:
        action=actions.FUNCTIONS.no_op()
    return action ,action_array



"""
    Runs 'agent' on map 'mapname' for 'iterations' iterations.
    Returns the data from all of the games run.
"""
def run_game_with_agent( agent, mapname, iterations):
    ### dqn parameters
    frame_num=4
    state_size = [84,84,7*frame_num]
    action_size = 32 +40+1            
    learning_rate =  0.001     

    
    eps_f = 0.05 
    eps_s = 1.00

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate

    ### TRAINING HYPERPARAMETERS
    total_episodes = 200   #prev 2000    # Total episodes for training
    batch_size = 10    #prev 100        
    iter_num = 10 #prev 20

    ### Experience HYPERPARAMETERS
    print("pre training!")
    pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
    experience_size = 3*batch_size #prev 800 
    dqn_sc2 = DQN(state_size, action_size, learning_rate)
    game_data = []
    exp= experience(experience_size) 
    decay_rate= 0.0005 #prev 0.0005
    with sc2_env.SC2Env(
        map_name=mapname,
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True),
        step_mul=100, #if too low, nothing really happens
        visualize=True,
        game_steps_per_episode=0) as env:

        agent.setup(env.observation_spec(), env.action_spec())
        #pretrain
        state = State(4)
        for i in range(pretrain_length):
            # print("Playing game {}".format(i+1))
            timesteps = env.reset()
            agent.reset()
            state.reinitialize( transform_state(timesteps) ) #initialize state
         
            while True :
                # print("new")
                state_old = state
                [step_actions ,step_actions_array] = random_step(timesteps[0])
                # print("random action: ", step_actions)
                done = False
                if timesteps[0].last():
                    done=True
                reward = get_reward(timesteps[0],done)
                timesteps = env.step([step_actions])
                state.add( transform_state(timesteps) )
                # for layer in state[0]:
                #     # print("layer: ", layer.shape)
                #     print ("Is zero?: ", (np.zeros(layer.shape) == layer).all())
                exp.add([state_old.get(),step_actions_array,  reward, state.get(), done])
                
                if timesteps[0].last():
                    break
                
                # print("timesteps [0] : ", type(timesteps[0]))

        #train
        agent.setup(env.observation_spec(), env.action_spec())
        saver = tf.train.Saver()
        # total_learning_episodes =10
        agent.reset()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            time = 0 #initialize time
            for i in range(iter_num):
                time=train_iteration(env, sess, agent,exp, saver, total_episodes,dqn_sc2, eps_s, eps_f,decay_rate,time,batch_size , gamma)
                test(env,  agent, saver,dqn_sc2,exp, eps_s, eps_f,decay_rate,time)
       

    # #execute with what we have
    #     print("testing!")
    #     with tf.Session() as sess:
    #         agent.setup(env.observation_spec(), env.action_spec())
            

    #         # Load the model
    #         saver.restore(sess, "./models/dqn_split.ckpt")
    #         state = State(4)
    #         for i in range(10):
    #             agent.reset()
    #             timesteps = env.reset()
                
    #             state.reinitialize(transform_state(timesteps) )
    #             totalScore = 0
    #             done=False
    #             while not done:
    #                 # Take the biggest Q value (= the best action)
                    
                    
    #                 Qs = sess.run(dqn_sc2.Yhat, feed_dict = {dqn_sc2.inputs_: state.get().reshape((1,*(state.get().shape) ))}) #run NN
                    
    #                 [step_actions , step_actions_array] =argmax_action(np.argmax(Qs),timesteps[0] )
    #                 state_old = state
    #                 timesteps = env.step([step_actions])
    #                 state.add(transform_state(timesteps))
                    
                    
    #                 if timesteps[0].last():
    #                     done=True
    #                 reward = get_reward(timesteps[0],done)
    #                 exp.add([state_old.get(),step_actions_array,  reward, state.get(), done])

                
    #     #             print("Score: ", reward)
    #                 totalScore += reward
    #             print("TOTAL_SCORE", totalScore)

    return []               
                

