#note to self . last time I finished learnig about the learning part and was about to go to execution. Also, I was thinking defining the pretrain, train, learning part as functions so I could overlapp them many times

"""
Written by Michael McGuire to provide an enhanced base agent with simple repetitive functions. (09 / 29 / 2018)
"""

import pysc2
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import app
import random
from dqn import *
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
    state_size = [84*84*5]
    action_size = 32 +40+1             # 2 * 4*4
    learning_rate =  0.001     

    
    eps_f = 0.05 
    eps_s = 1.00

    # Q learning hyperparameters
    gamma = 0.95               # Discounting rate

    ### TRAINING HYPERPARAMETERS
    total_episodes = 100  #prev 800      # Total episodes for training
    max_steps = 100              # Max possible steps in an episode
    batch_size = 20    #prev 60 I think       

    ### Experience HYPERPARAMETERS
    print("pre training!")
    pretrain_length = batch_size   # Number of experiences stored in the Memory when initialized for the first time
    experience_size = 500 
    dqn_sc2 = DQN(state_size, action_size, learning_rate)
    game_data = []
    exp= experience(experience_size) 
    decay_rate=0.01
    with sc2_env.SC2Env(
        map_name=mapname,
        agent_interface_format=features.AgentInterfaceFormat(
            feature_dimensions=features.Dimensions(screen=84, minimap=64),
            use_feature_units=True),
        step_mul=20, #if too low, nothing really happens
        visualize=True,
        game_steps_per_episode=0) as env:

        agent.setup(env.observation_spec(), env.action_spec())
        #pretrain
        for i in range(pretrain_length):
            # print("Playing game {}".format(i+1))
            timesteps = env.reset()
            agent.reset()
            state = transform_state(timesteps)
                                
            while True:
                # print("new")
                state_old = state
                [step_actions ,step_actions_array] = random_step(timesteps[0])
                # print("random action: ", step_actions)
                done = False
                if timesteps[0].last():
                    done=True
                reward = get_reward(timesteps[0],done)
                timesteps = env.step([step_actions])
                state = transform_state(timesteps)
                # for layer in state[0]:
                #     # print("layer: ", layer.shape)
                #     print ("Is zero?: ", (np.zeros(layer.shape) == layer).all())
                exp.add([state_old,step_actions_array,  reward, state, done])
                if timesteps[0].last():
                    break
                
                # print("timesteps [0] : ", type(timesteps[0]))

        #train
        agent.setup(env.observation_spec(), env.action_spec())
        saver = tf.train.Saver()
        total_learning_episodes =10
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            time = 0 #initialize time
            _=learn(env, sess, exp, saver,batch_size ,total_episodes ,dqn_sc2, gamma)
            time=train(env, sess, agent,exp, total_episodes,dqn_sc2, eps_s, eps_f,decay_rate,time )
            _=learn(env, sess, exp, saver,batch_size ,total_episodes ,dqn_sc2, gamma)
        #     for episode in range(total_episodes):
        #         timesteps = env.reset()
        #         agent.reset()
                

        #         # Make a new episode and observe the first state
        #         timesteps = env.reset()
        #         agent.reset()
        #         state = transform_state(timesteps)
        #         time=0
        #         while True:
        #             state_old = state
        #             [step_actions , step_actions_array] = eps_decay(state,sess,dqn_sc2, eps_s, eps_f,decay_rate,time ,timesteps[0])
        #             # print("eps action: ", step_actions)
        #             done = False
        #             if timesteps[0].last():
        #                 done=True
        #             reward = get_reward(timesteps[0],done)
        #             timesteps = env.step([step_actions])
        #             state = transform_state(timesteps)
        #             exp.add([state_old,step_actions_array,  reward, state, done])
        #             if timesteps[0].last():
        #                 break
        #             time+=1 #increase time for eps_decay
            
        #         #learning part         
        #         # Obtain random mini-batch from memory
        #         print("learning!")
        #         batch = exp.sample(batch_size)
        #         states_mb=[]
        #         actions_mb=[]
        #         rewards_mb=[]
        #         next_states_mb=[]
        #         dones_mb=[]
                
        #         for i in batch:
        # #                 print("states: ",i[0])
        #             states_mb.append(i[0])
        #             actions_mb.append(i[1] )
        #             rewards_mb.append(i[2])
        #             next_states_mb.append(i[3])
        #             dones_mb.append(i[4])
        # #             states_mb = np.array([each[0] for each in batch], ndmin=3)
        # #             actions_mb = np.array([each[1] for each in batch])
        # #             rewards_mb = np.array([each[2] for each in batch]) 
        # #             dones_mb = np.array([each[4] for each in batch])
        # #             next_states_mb = np.array([each[3] for each in batch], ndmin=3)
        #         # print("states_mb: ",states_mb )
                
        #         states_mb=np.array(states_mb)
        #         states_mb=np.squeeze(states_mb)
        #         actions_mb=np.array(actions_mb)
        #         actions_mb=np.atleast_2d(actions_mb)
        #         # actions_mb=np.squeeze(actions_mb) 
        #         rewards_mb=np.array(rewards_mb)
        #         next_states_mb=np.array(next_states_mb)
        #         next_states_mb=np.squeeze(next_states_mb)
        #         dones_mb=np.array(dones_mb)
        #         # print("actions_mb[0]: ", (actions_mb[0] ).shape)
        #         # print("actions_mb[1]: ",len(actions_mb[1]) )
                
        # #             print("action_mb shape: ",actions_mb.shape)
        # #             print("states_mb shape: ",states_mb.shape)
        # #             print("rewards_mb shape: ",rewards_mb.shape)
                
        #         target_Qs_batch = []

        #         # Get Q values for next_state 
        #         Qs_next_state = sess.run(dqn_sc2.Yhat, feed_dict = {dqn_sc2.inputs_: next_states_mb})

        #         # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        #         for i in range(0, batch_size):
        #             terminal = dones_mb[i]
        #             # If we are in a terminal state, only equals reward
        #             if terminal:
        #                 target_Qs_batch.append(rewards_mb[i])
        #             else:
        #                 target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
        #                 target_Qs_batch.append(target)

        #         targets_mb = np.array([each for each in target_Qs_batch])
                
                
        #         sess.run( dqn_sc2.optimizer, #what happens when theres two inputs
        #                             feed_dict={dqn_sc2.inputs_: states_mb,
        #                                         dqn_sc2.target_Q: targets_mb,
        #                                         dqn_sc2.actions_: actions_mb})
            
        #         # Save model every 5 episodes
        #         if episode % 5 == 0:
        #             save_path = saver.save(sess, "./models/dqn_split.ckpt")
        #             print("Model Saved")

    #execute with what we have
        print("testing!")
        with tf.Session() as sess:
            agent.setup(env.observation_spec(), env.action_spec())
            

            # Load the model
            saver.restore(sess, "./models/dqn_split.ckpt")
            for i in range(10):
                agent.reset()
                timesteps = env.reset()
                
                state = transform_state(timesteps)
                totalScore = 0
                done=False
                while not done:
                    # Take the biggest Q value (= the best action)
                    
                    
                    Qs = sess.run(dqn_sc2.Yhat, feed_dict = {dqn_sc2.inputs_: state}) #run NN
                    
                    [step_actions , step_actions_array] =argmax_action(np.argmax(Qs),timesteps[0] )
                    state_old = state
                    timesteps = env.step([step_actions])
                    state = transform_state(timesteps)
                    
                    
                    if timesteps[0].last():
                        done=True
                    reward = get_reward(timesteps[0],done)
                    exp.add([state_old,step_actions_array,  reward, state, done])

                
        #             print("Score: ", reward)
                    totalScore += reward
                print("TOTAL_SCORE", totalScore)

    return []               
                

