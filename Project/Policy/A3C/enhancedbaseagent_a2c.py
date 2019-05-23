
"""
Written by Michael McGuire to provide an enhanced base agent with simple repetitive functions. (09 / 29 / 2018)
"""

import pysc2
from pysc2.lib import actions, features, units
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from absl import app
import random
from a2c import *
import numpy as np
import tensorflow as tf

class TerranAgent(base_agent.BaseAgent):
    def __init__(self):
        super(TerranAgent, self).__init__()
        self.attack_coordinates = None


def run_game(unused_argv ):
    '''plays games under pysc2 env'''
    ### TRAINING HYPERPARAMETERS
    total_episodes = 1   #prev 200    # Total episodes for training       
    iter_num = 10 #prev 20
    mapname="DefeatZerglingsAndBanelings"
    #initialization of hyperparameters
    screen_size=(17,84,84)
    minimap_size=(7,84,84)
    non_spatial_size = [11]
    non_spatial_action_size= [4]
    spatial_action_size = (84,84)
    learning_rate = 0.00000045

    a2c = NN(screen_size, minimap_size, non_spatial_size, non_spatial_action_size,spatial_action_size,  learning_rate,  name = "a2c")
    discount_rate= 0.0000009 #prev 0.0005
    batch_size = 4 # random number, I am yet to know what this is for
    agent = TerranAgent()
    try:
        with sc2_env.SC2Env(
            map_name=mapname,
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=84, minimap=64),
                use_feature_units=True),
            step_mul=20, 
            visualize=True,
            game_steps_per_episode=0) as env:
            saver = tf.train.Saver()
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                agent.setup(env.observation_spec(), env.action_spec())
                
                # time = 0 #initialize time
                #memory = test( sess, agent, a2c, env, screen_size, minimap_size, non_spatial_size, non_spatial_action_size,spatial_action_size, total_episodes)
                #train( sess,agent, memory, discount_rate, a2c, batch_size, screen_size, minimap_size, non_spatial_size, non_spatial_action_size,spatial_action_size, total_episodes)
                
                for i in range(iter_num):
                    done=False
                    timesteps = env.reset()
                    agent.reset()
                    while not done: 
                        screen_obs, minimap_obs, non_spatial_obs = transform_obs(timesteps)
                        feed_dict ={a2c.screen_input_: screen_obs, a2c.minimap_input_: minimap_obs , a2c.non_spatial_input_: non_spatial_obs }
                        conv1, conv2, conv_shape, screen, minimap , space , flatten= sess.run([a2c.screen_conv1_out, a2c.screen_conv2_out, a2c.state_representation_, a2c.screen_conv2_out ,a2c.minimap_conv2_out, a2c.primary_action_spatial_, a2c.flatten], feed_dict = feed_dict)
                        print("conv1 shape: ",conv1.shape)
                        print("conv2 shape: ",conv2.shape)
                        print("state shape: ",conv_shape.shape)
                        print("screen sahpe: ", screen.shape)
                        print("minmap shape: ", minimap.shape)
                        print("space shape: ", space.shape)
                        print("flatten shape: ", flatten.shape)
                        done= True
                        test( sess, NN, env, state_size, non_spatial_action_size, spatial_action_size, total_episodes)
                        timesteps = env.step([action])
                        if timesteps[0].last():
                            done=True
                    
  
    except KeyboardInterrupt:
        pass
    return []               



if __name__ == "__main__":
    app.run(run_game)
                

