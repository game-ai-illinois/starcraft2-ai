###############################################################################

#
#   File Created: 25th Thurs 2018
#   Author: Hyeon-Seo Yun
#   File: a2c.py
#   Path: 
#   References: 
#      https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/model.py
#   
#

###############################################################################

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import tensorflow as tf
import random
from collections import deque

def transform_obs(timesteps):
    '''takes in the game observation and returns compatible input for neural network'''
    screen_state =timesteps[0].observation.feature_screen.reshape((1,*timesteps[0].observation.feature_screen.shape[::-1]))
    # print("is selected present?: ", timesteps[0].observation.feature_screen.selected in timesteps[0].observation.feature_screen)
    # print("selected shape: ", timesteps[0].observation.feature_screen.selected.shape, timesteps[0].observation.feature_screen.shape)
    # print("selected: ", np.array_equal(timesteps[0].observation.feature_screen.selected, np.zeros((84,84))))
    minimap_state = np.zeros((1,84,84,7)) #match minimap size to screen size -> (7,84,84)
    minimap_state[:,10:74,10:74,:] = timesteps[0].observation.feature_minimap.reshape((64,64,7)) # add in the minimap input, which has shape of (7,64,64)
    non_spatial_state =timesteps[0].observation.player.reshape((1,*timesteps[0].observation.player.shape[::-1]))
    # print("feature screen: ",timesteps[0].observation.feature_screen.shape)
    # print("feature mini map: ",timesteps[0].observation.feature_minimap.shape)
    # print("non-spatial features: ",timesteps[0].observation.player)

    return (screen_state, minimap_state,non_spatial_state)


def one_hot(arg, size):
    arr = np.zeros((size))
    arr[arg]= 1.0
    return arr

"""
    Selects all units of type unit_type 
"""
def select_units_by_type( obs, unit_type):
    
    my_units = get_units_by_type(obs, unit_type)
    if (len(my_units) > 0):
        u = random.choice(my_units)
        return actions.FUNCTIONS.select_point("select_all_type", (u.x, u.y))



"""
    Returns list of units of type unit_type in the feature_units layer of obs
"""
def get_units_by_type( obs, unit_type):
    return [unit for unit in obs.observation.feature_units if unit.unit_type == unit_type]

def get_reward( timesteps, done) :
    '''observes the screen and calculates the immediate reward'''
    obs = timesteps[0]
    reward = -0.01 #living cost
    # my_units = get_units_by_type(obs,units.Terran.Marine)
    # zergling  = get_units_by_type(obs,units.Zerg.Zergling)
    # baneling  = get_units_by_type(obs,units.Zerg.Baneling)
    # living_cost =10
    # ally_coef = 0.1
    # if not done:
    #     reward=  ally_coef*len(my_units) - len(zergling) -len(baneling) - living_cost#extra minus 10 for living
    #     # return len(my_units) - len(zergling) -len(baneling) -10 #extra minus 10 for living
    # else:
    #     # print("last step")
    #     if len(zergling) +len(baneling) ==0: #if you won the game
    #         reward = ally_coef*len(my_units) - len(zergling) -len(baneling) -living_cost + 1000 #add big reward for winning
    #         print("we won!")
    #         # return len(my_units) - len(zergling) -len(baneling) -10 + 1000 #add big reward for winning
    #     else:
    #         reward = ally_coef*len(my_units) - len(zergling) -len(baneling) -living_cost 
    #         # return len(my_units) - len(zergling) -len(baneling) -10 - 1000 #add big punishment for losing
    if obs.last():
        base_reward = 100 #this exists to encourage an agent to end the game. If it does nothing, the game will end without last observation triggering\
        #thus making the reward for that game zero
        my_units = get_units_by_type(obs,units.Terran.Marine)
        zergling  = get_units_by_type(obs,units.Zerg.Zergling)
        baneling  = get_units_by_type(obs,units.Zerg.Baneling)
        # reward += obs.reward
        reward += 0.25*(len(my_units)) - len(zergling) -len(baneling) +base_reward
        print("reward: ", reward)
    return np.array([reward])


def trans_action(action_arg , array_size =32+40+1): # action is assumed to be the arg value of the action layer
    return_array = np.zeros(array_size)
    return_array[action_arg] =1
    return return_array

def random_2d_choice(input):
    matrix = input.reshape((84,84))
    # print(matrix.shape)
    # print(np.sum(matrix))
    array = np.sum(matrix, axis=1)
    yarg= np.random.choice(len(array),1, p=array)
    # print(matrix[yarg].shape)
    xarg = np.random.choice(len(matrix[0]),1, p=matrix[yarg].reshape(84)/ np.sum(matrix[yarg]))
    return [xarg, yarg]

def can_do( obs, action):
    return action in obs[0].observation.available_actions

def action(non_spatial_policy, primary_spatial_policy, obs) :
    '''Takes in the action probability distribution and returns pysc2 action and deterministic actons
        result = (action variable for pysc2, one hot array of non spatial actino, )    
    '''
    flag = "now"
    # flag = "queued"
    #non_spatial_policy_list = ["do nothing", "attack at", "move at", "patrol", "hold position", "stop"]
    # print(non_spatial_policy.shape, primary_spatial_policy.shape,  (is_primary))
    non_spatial_policy_size = 7
    non_spatial_policy = non_spatial_policy.reshape((non_spatial_policy_size))
    action_non_spatial = np.random.choice(len(non_spatial_policy),1, p=non_spatial_policy)
    _ , screen_y_len, screen_x_len, _ = primary_spatial_policy.shape
    # print("action: ", action_non_spatial)
    if action_non_spatial == 0: #do nothing
        result = [actions.FUNCTIONS.no_op() ,\
         one_hot(action_non_spatial,non_spatial_policy_size) , np.zeros((screen_y_len,screen_x_len)) ,(False)] 
         
    elif action_non_spatial == 1: #attack at
        x,y = random_2d_choice(primary_spatial_policy)
        primary_determine = np.zeros((screen_y_len,screen_x_len))
        primary_determine[y ,x ] =1.0 # row number is y, column number is x
        if can_do(obs, actions.FUNCTIONS.Attack_screen.id):
            result = [actions.FUNCTIONS.Attack_screen(flag, (x,y )) , \
                 one_hot(action_non_spatial,non_spatial_policy_size) ,  primary_determine , (True)]
        else: #if  not available, do nothing
            print("action not available")
            result = [actions.FUNCTIONS.no_op() , \
                one_hot(action_non_spatial,non_spatial_policy_size), primary_determine ,(True)] 
    elif action_non_spatial == 2:    # move at
        x,y = random_2d_choice(primary_spatial_policy)
        primary_determine = np.zeros((screen_y_len,screen_x_len))
        primary_determine[y ,x ] =1.0 # row number is y, column number is x
        if can_do(obs, actions.FUNCTIONS.Move_screen.id):
            result = [actions.FUNCTIONS.Move_screen(flag, (x,y )) ,\
                 one_hot(action_non_spatial,non_spatial_policy_size) ,  primary_determine , (True)]
        else: #if  not available, do nothing
            print("action not available")
            result = [actions.FUNCTIONS.no_op() , \
                one_hot(action_non_spatial,non_spatial_policy_size)  , primary_determine , (True)] 
    elif action_non_spatial == 3:    # patrol
        x,y = random_2d_choice(primary_spatial_policy)
        primary_determine = np.zeros((screen_y_len,screen_x_len))
        primary_determine[y ,x ] =1.0 # row number is y, column number is x
        if can_do(obs, actions.FUNCTIONS.Patrol_screen.id):
            result = [actions.FUNCTIONS.Patrol_screen(flag, (x,y )) , \
                one_hot(action_non_spatial,non_spatial_policy_size) ,  primary_determine , (True)]
        else: #if  not available, do nothing
            print("action not available")
            result = [actions.FUNCTIONS.no_op() , \
                one_hot(action_non_spatial,non_spatial_policy_size)  , primary_determine , (True)] 
    elif action_non_spatial == 4:    # hold
        if can_do(obs, actions.FUNCTIONS.HoldPosition_quick.id):
            result = [actions.FUNCTIONS.HoldPosition_quick(flag) , \
                one_hot(action_non_spatial,non_spatial_policy_size) , np.zeros((screen_y_len,screen_x_len)), (False)]
        else: #if  not available, do nothing
            print("action not available")
            result = [actions.FUNCTIONS.no_op() ,\
                 one_hot(action_non_spatial,non_spatial_policy_size)  , np.zeros((screen_y_len,screen_x_len)), (False)] 
    else:    # stop
        if can_do(obs, actions.FUNCTIONS.Stop_quick.id):
            result = [actions.FUNCTIONS.Stop_quick(flag) , \
                one_hot(action_non_spatial,non_spatial_policy_size) , np.zeros((screen_y_len,screen_x_len)),(False)]
        else: #if  not available, do nothing
            print("action not available")
            result = [actions.FUNCTIONS.no_op() ,\
                 one_hot(action_non_spatial,non_spatial_policy_size)  , np.zeros((screen_y_len,screen_x_len)), (False)] 
    
    return result

# def return_state(timestep):
#         obs = timestep[0]
#         screen_obs.observation["feature_screen"]
#         minimap_obs = obs.observation["feature_minimap"]
#         scr =   obs.observation.feature_screen
#         # unit_type = obs.observation["feature_screen"][units.Terran.Marine]
#         _UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
#         # scipy.misc.imsave('test.jpg', (( obs.observation.feature_screen.unit_type == units.Terran.Marine)).astype(int) ) #105 is idex for zergling

#         #computes array of locations of enemy zerglings
#         hostile_zerglings =(np.array(scr[_UNIT_TYPE] ) == 105).astype(int) 

#         #computes array of locations of enemy banelings
#         hostile_banelings =(np.array(scr[_UNIT_TYPE] ) == 9).astype(int) 

#         ### Computes array of locations of selected marines
#         friendly_selected = np.array(scr.selected)
#         # scipy.misc.imsave('test.jpg', friendly_selected ) 
#         ### Computes arrays of locations of marines and enemy units
#         # player_relative = np.array(scr.player_relative)
#         # player_friendly = (player_relative == _PLAYER_FRIENDLY).astype(int)
#         # player_hostile = (player_relative == _PLAYER_HOSTILE).astype(int)
        
#         ### Computes arrays of hitpoints for marines and enemy units
#         player_hitpoints = np.array(scr.unit_hit_points) 
#         marine_hitpoints = player_hitpoints * player_friendly
#         zergling_hitpoints = player_hitpoints * hostile_zerglings
#         baneling_hitpoints = player_hitpoints * hostile_banelings
#         #baneling_hitpoints = np.multiply(player_hitpoints, player_hostile)
        
#         ### Stacks the previous arrays in the order given in the documentation. This will be the primary input to the neural network.
#         array = np.vstack([friendly_selected, marine_hitpoints, zergling_hitpoints, baneling_hitpoints,  hostile_zerglings, hostile_banelings])
#         #array = np.stack([friendly_selected, friendly_hitpoints, friendly_density, hostile_hitpoints, hostile_density, hostile_zerglings, hostile_banelings], axis=0)
        
#         # ------------------------------------- #        
        
#         ### Computes the number of marines that are still alive
#         army_count = len(obs.observation.feature_units)
        
#         zergling_count = np.sum(hostile_zerglings) 
#         baneling_count = np.sum(hostile_banelings)
#         enemy_count = zergling_count +baneling_count #adding this sum so it may give a easier correlation for the AI to find out in the intial phase of training

#         return (array, [army_count, enemy_count, zergling_count, baneling_count])




def softmax(logits, name):
    return tf.exp(logits, name = name) / tf.reduce_sum(tf.exp(logits, name= name), name=name)

class NN:
    def __init__(self, screen_size, minimap_size, non_spatial_size, non_spatial_action_size, spatial_action_size,  learning_rate, name):
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.non_spatial_size=non_spatial_size
        self.non_spatial_action_size = non_spatial_action_size
        self.spatial_action_size = spatial_action_size
        self.learning_rate = learning_rate
        self.name = name
        # self.pdtype = make_pdtype(action_space)
        self.entropy_coef =0.01 # assuming entropy penalty means entropy coeff
        # self.value_coef = 0.5
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        #format of the NN heavily taken by https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/A2C%20with%20Sonic%20the%20Hedgehog/model.py
        # primary action space is used to move to , attack, patrol actions
        with tf.variable_scope(self.name):
            self.screen_input_= tf.placeholder(tf.float32, shape=[None, *self.screen_size[::-1]], name="screen_input")
            self.minimap_input_= tf.placeholder(tf.float32, shape=[None, *self.minimap_size[::-1]], name="minimap_input")
            self.non_spatial_input_= tf.placeholder(tf.float32, shape=[None, self.non_spatial_size], name="non_spatial_input")
            self.non_spatial_actions_= tf.placeholder(tf.float32, shape=[None, *non_spatial_action_size], name='non_spatial_actions_')
            self.advantages_non_spatial_= tf.placeholder(tf.float32, shape=[None, 1], name='advantages_non_spatial')
            self.primary_actions_= tf.placeholder(tf.float32, shape=[None, *spatial_action_size], name='primary_actions_')
            self.rewards_  = tf.placeholder(tf.float32, [None,1], name="rewards")
            self.is_primary = tf.placeholder(tf.float32, [None,1], name="is_primary") #indicates if primary spatial action space was used
            
            #convolution layer for feature screen
            # Screen Input is 4x17x84x84
            self.screen_conv1 = tf.layers.conv2d(inputs = self.screen_input_,
                                            filters = 16,
                                            kernel_size = [5,5],
                                            strides = [1,1], 
                                            padding = "same",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name = "screen_conv1")
            
            self.screen_conv1_batchnorm = tf.layers.batch_normalization(self.screen_conv1,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'screen_conv1_batchnorm')
            
            self.screen_conv1_out = tf.nn.relu(self.screen_conv1_batchnorm, name="screen_conv1_out")
            ## --> [dunno]
            
            
            self.screen_conv2 = tf.layers.conv2d(inputs = self.screen_conv1_out,
                                    filters = 32,
                                    kernel_size = [3,3],
                                    strides = [1,1],
                                    padding = "same",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    name = "screen_conv2")
        
            self.screen_conv2_batchnorm = tf.layers.batch_normalization(self.screen_conv2,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'screen_conv2_batchnorm')

            self.screen_conv2_out = tf.nn.relu(self.screen_conv2_batchnorm, name="screen_conv2_out")
            ## --> [dunno]
            
            

            #convolution layer for minimap
            # Screen Input is 4x7x84x84
            self.minimap_conv1 = tf.layers.conv2d(inputs = self.minimap_input_,
                                            filters = 16,
                                            kernel_size = [5,5],
                                            strides = [1,1],
                                            padding = "same",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name = "minimap_conv1")
            
            self.minimap_conv1_batchnorm = tf.layers.batch_normalization(self.minimap_conv1,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'minimap_conv1_batchnorm')
            
            self.minimap_conv1_out = tf.nn.relu(self.minimap_conv1_batchnorm, name="minimap_conv1_out")
            ## --> [dunno]
            
            
            
            self.minimap_conv2 = tf.layers.conv2d(inputs = self.minimap_conv1_out,
                                    filters = 32,
                                    kernel_size = [3,3],
                                    strides = [1,1],
                                    padding = "same",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    name = "minimap_conv2")
        
            self.minimap_conv2_batchnorm = tf.layers.batch_normalization(self.minimap_conv2,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'minimap_conv2_batchnorm')

            self.minimap_conv2_out = tf.nn.relu(self.minimap_conv2_batchnorm, name="minimap_conv2_out")
            ## --> [dunno]
            
            #stack all the input into one big state representation
            # tf.print(tf.shape(self.screen_conv2_out) )
            # tf.print(tf.shape(self.screen_conv2_out) )
            
            # self.non_spatial_obs_out = tf.concat(self.non_spatial_input_ ,tf.zeros([84*84-7], tf.float32))
            # tf.zeros([84, 84], tf.float32)
            # self.non_spatial_obs_out[:self.non_spatial_size[0]] = self.non_spatial_input_ #turn non spatial to 84x84

            self.state_representation_ = tf.concat([self.screen_conv2_out, self.minimap_conv2_out], axis=3) #don't forget to add in the non spatial input as well
            # self.state_representation_ = tf.cast(self.state_representation_, dtype = tf.float32) # cast it as float so conv2d won't give an issue
            #spatial action policy
            self.primary_conv = tf.layers.conv2d(inputs = self.state_representation_,
                                    filters = 1,
                                    kernel_size = [1,1],
                                    strides = [1,1],
                                    padding = "same",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    activation=None,
                                    name = "primary_conv")
            self.primary_action_conv_ =softmax(self.primary_conv, name = "primary_action_conv_")
            
        
            #non-spatial action policy
            self.flatten = tf.concat([tf.layers.flatten(self.state_representation_), self.non_spatial_input_ ], axis = 1)
            self.hiddenlayer_common_non_spatial = tf.layers.dense( self.flatten , units=256, activation = tf.nn.relu) 

            self.value = tf.layers.dense( self.hiddenlayer_common_non_spatial ,kernel_initializer=tf.contrib.layers.xavier_initializer(), units=1, activation = None)  #creates value function
            self.policy =tf.layers.dense( self.hiddenlayer_common_non_spatial ,kernel_initializer=tf.contrib.layers.xavier_initializer(), units=self.non_spatial_action_size[0], activation = tf.nn.softmax)

            
            C = 1e-8
            self.entropy_non_spatial = -tf.reduce_sum(self.policy * tf.log(C + self.policy)) 
            self.entropy_spatial_primary = -tf.reduce_sum(self.primary_action_conv_ * tf.log(C + self.primary_action_conv_)) 
            #policy loss = 1/n * sum A(si,ai) * -logpi(ai|si)
            self.policy_loss_non_spatial = tf.reduce_sum(-tf.log(C+ tf.reduce_sum(self.policy*  tf.cast(self.non_spatial_actions_, tf.float32), [1]))*self.advantages_non_spatial_)
            # Value loss 1/2 SUM [R - V(s)]^2
            self.value_loss_non_spatial = 0.5 *tf.reduce_sum(tf.square(self.rewards_ - self.value))

            #adding spatial policy loss
            self.advantages_spatial_ = tf.reduce_sum(self.advantages_non_spatial_ , axis =1 ) # a reshaped advantages_non_spatial_ tensor

            #reshape to match actions variables
            self.primary_action_conv_out_ = tf.reduce_sum(self.primary_action_conv_, axis=-1)
            self.policy_loss_spatial_primary = tf.reduce_sum(-tf.log(C+ tf.reduce_sum(self.primary_action_conv_out_ * self.primary_actions_))*self.advantages_spatial_) 

            self.loss_noop = self.value_loss_non_spatial + self.policy_loss_non_spatial + self.entropy_non_spatial * self.entropy_coef
            self.loss_primary =    self.policy_loss_spatial_primary + self.entropy_spatial_primary   *self.entropy_coef

            
            
            self.loss_total = self.loss_noop + self.is_primary * self.loss_primary 
            # self.loss_total = - self.loss_total
            
            # self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_total)
            variable = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            self.gradients= tf.gradients(self.loss_total,variable)
            grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,40.0)
            self.apply_grads = self.trainer.apply_gradients(zip(grads,variable))



 
#update target graph implementation from https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
# def update_target_graph():
    
#     # Get the parameters of our dqn_sc2
#     from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn_sc2")
    
#     # Get the parameters of our dqn_target
#     to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn_target")

    
#     # Update our target_network parameters with DQNNetwork parameters
#     for from_var,to_var in zip(from_vars,to_vars):
#         to_var.assign(from_var)
    

def test( sess, agent, NN, env, screen_size, minimap_size, non_spatial_size, non_spatial_action_size,spatial_action_size, total_episodes):
    '''play the game for total_episodes many times and return memory to train from'''
    '''memory = dict of lists where each element is np array'''
    memory={} # each element [s,a,s',r,v]
    memory["screen_state"] = []
    memory["minimap_state"] = []
    memory["non_space_state"] = []
    memory["next_screen_state"] = []
    memory["next_minimap_state"] = []
    memory["next_non_space_state"] = []
    memory["action_non_spatial"] = []
    memory["action_spatial_primary"] = []
    memory["reward"] = []
    memory["value"] = []
    memory["is_primary"] = []
    total_reward = 0
    f= open("average_reward.csv","a+")
    for episode in range(total_episodes):
        timesteps_current = env.reset()
        agent.reset()
        done = False
        timer = 0 # add a time limit to each episode
        timer_limit =100
        #now initialize stacks to list memory frames
        screen_state_stack = np.empty((0  ,*screen_size[::-1]), float)
        minimap_state_stack = np.empty((0  ,*minimap_size[::-1]), float)
        non_space_state_stack = np.empty((0  ,non_spatial_size), float)
        next_screen_state_stack = np.empty((0  ,*screen_size[::-1]), float)
        next_minimap_state_stack = np.empty((0  ,*minimap_size[::-1]), float)
        next_non_space_state_stack = np.empty((0  ,non_spatial_size), float)
        action_non_spatial_stack =  np.empty((0  ,*non_spatial_action_size), float)
        action_spatial_primary_stack = np.empty((0  ,*spatial_action_size), float)
        reward_stack= np.empty((0  ,1), float)
        value_stack = np.empty((0  ,1), float)
        is_primary_stack = np.empty((0  ,1), float)
        
        #start playing the game
        screen_obs_queue = deque(maxlen=4)
        minimap_obs_queue =  deque(maxlen=4)
        non_spatial_obs_queue = deque(maxlen=4)
        screen_obs, minimap_obs, non_spatial_obs = transform_obs(timesteps_current) #note: the obs shapes are (1,84,84,17), (1,84,84,7), (1,7)
        # print("screen_obs sahpe: ", screen_obs.shape)
        for i in range(4): #first time, intialize the queue with the same shot
            screen_obs_queue.append(screen_obs)
            minimap_obs_queue.append(minimap_obs)
            non_spatial_obs_queue.append(non_spatial_obs)

        while (not done) and timer < timer_limit:
            screen_state =np.array(screen_obs_queue).reshape((1,*screen_size[::-1]))
            # print("screen_state shape : ", screen_state.shape)
            minmap_state =  np.array(minimap_obs_queue).reshape((1,*minimap_size[::-1]))
            non_spatial_state =np.array(non_spatial_obs_queue).reshape((1,non_spatial_size))
            # print("screen_state_stack shape: ", screen_state_stack.shape)
            screen_state_stack = np.vstack((screen_state_stack, screen_state))
            minimap_state_stack = np.vstack((minimap_state_stack, minmap_state))
            non_space_state_stack = np.vstack((non_space_state_stack, non_spatial_state))

            feed_dict ={NN.screen_input_: screen_state, NN.minimap_input_: minmap_state , NN.non_spatial_input_: non_spatial_state}
            non_spatial_policy, p_spatial_policy,  value = sess.run([NN.policy, NN.primary_action_conv_,\
              NN.value], feed_dict=feed_dict )
            # print("spatial policy sum: ", np.sum(p_spatial_policy))
            # print("spatial policy : ", (p_spatial_policy))

            
            [action_step, non_spatial_determine , primary_spatial_determine , is_tuple ] = action(\
                non_spatial_policy, p_spatial_policy, timesteps_current) #is_tuple = (is_primary)
            # print("non_spatial_determine shape: ", non_spatial_determine.shape)
            # print("primary_spatial_determine shape: ", primary_spatial_determine.shape)
            timesteps_next = env.step([action_step])
            screen_obs, minimap_obs, non_spatial_obs = transform_obs(timesteps_next)
            screen_obs_queue.append(screen_obs)
            minimap_obs_queue.append(minimap_obs)
            non_spatial_obs_queue.append(non_spatial_obs)

            if timesteps_next[0].last():
                print("LAst step REward: ", timesteps_next[0].reward)
                done=True
                print("done")
            reward = get_reward(timesteps_next, done)
            total_reward += reward
            next_screen_state =np.array(screen_obs_queue).reshape((1,*screen_size[::-1]))
            next_minmap_state =  np.array(minimap_obs_queue).reshape((1,*minimap_size[::-1]))
            next_non_spatial_state =np.array(non_spatial_obs_queue).reshape((1, non_spatial_size))

            next_screen_state_stack = np.vstack((next_screen_state_stack, next_screen_state))
            next_minimap_state_stack = np.vstack((next_minimap_state_stack, next_minmap_state))
            next_non_space_state_stack = np.vstack((next_non_space_state_stack, next_non_spatial_state))
            action_non_spatial_stack = np.vstack((action_non_spatial_stack, non_spatial_determine.reshape(1 , *non_spatial_determine.shape )))
            action_spatial_primary_stack = np.vstack((action_spatial_primary_stack, primary_spatial_determine.reshape(1 , *primary_spatial_determine.shape )))
            reward_stack = np.vstack((reward_stack, reward.reshape(1 , 1 )))
            value_stack = np.vstack((value_stack, value.reshape(1,1)))
            is_primary_stack = np.vstack((is_primary_stack,is_tuple))
            timesteps_current = timesteps_next
            timer +=1

        memory["screen_state"].append(screen_state_stack)
        memory["minimap_state"].append(minimap_state_stack)
        memory["non_space_state"].append(non_space_state_stack)
        memory["next_screen_state"].append(next_screen_state_stack)
        memory["next_minimap_state"].append(next_minimap_state_stack)
        memory["next_non_space_state"].append(next_non_space_state_stack)
        memory["action_non_spatial"].append(action_non_spatial_stack)
        memory["action_spatial_primary"].append(action_spatial_primary_stack)
        memory["reward"].append(reward_stack)
        memory["value"].append(value_stack)
        memory["is_primary"].append(is_primary_stack)

    # print("memory state ",memory["state"].shape)
    avg_reward = total_reward /total_episodes
    print("average reward, ", avg_reward)
    f.write(str(avg_reward[0])+" \n" )   
    print(type(avg_reward[0]))
    print("wrote some stuff")
    return (memory)

def train( sess,agent, memory, discount_rate, NN, batch_size, \
screen_size, minimap_size, non_spatial_size, non_spatial_action_size,spatial_action_size): 
    ''' trains the neural network with the given memory variable '''
    f_loss = open("average_loss.csv","a+")
    # print("memory['value'] : ", memory["value"])
    # print("batch_frame_len : ",batch_frame_len)
    #initialize batch variables
    screen_batch = np.empty((0,*screen_size[:: -1]),float)
    minimap_batch = np.empty((0,*minimap_size[:: -1]),float)
    non_spatial_batch = np.empty((0,non_spatial_size),float)
    next_screen_batch = np.empty((0,*screen_size[:: -1]),float)
    next_minimap_batch = np.empty((0,*minimap_size[:: -1]),float)
    next_non_spatial_batch = np.empty((0,non_spatial_size),float)
    non_spatial_action_batch =  np.empty((0,*non_spatial_action_size),float)
    primary_spatial_action_batch =  np.empty((0,*spatial_action_size),float)
    reward_batch =   np.empty((0,1),float)
    current_value_batch =  np.empty((0,1),float)
    is_primary_batch =  np.empty((0,1),float)
    
    #fill in the batches with memory clips
    memory_len = len(memory["value"]) # no particular reason I picked value key, just to get the length 
    batch_frame_len = 20 # number of frames of each memory sequence

        
    for iteration in range(batch_size):
        i = random.randrange(0, memory_len)
        if len(memory["value"][i]) < 20:
            batch_frame_len =len(memory["value"][i]) 
        print("length of memory value : " ,len(memory["value"][i])  )
        print("rand range limit: ",  len(memory["value"][i]) + 1- batch_frame_len )
        j = random.randrange(0,  len(memory["value"][i]) + 1- batch_frame_len  )
        screen_memory_clip = memory["screen_state"][i][j:j+batch_frame_len]
        # print("J: ", j)
        # print("screen_memory_clip shape: ",screen_memory_clip.shape)
        # print("screen_batch shape: ",screen_batch.shape)
        
        screen_batch = np.vstack((screen_batch, screen_memory_clip.reshape(( *screen_memory_clip.shape)) ))
        minimap_memory_clip = memory["minimap_state"][i][j:j+batch_frame_len]
        minimap_batch = np.vstack((minimap_batch, minimap_memory_clip.reshape(( *minimap_memory_clip.shape)) ))
        
        non_spatial_memory_clip = memory["non_space_state"][i][j:j+batch_frame_len]
        # print(" memory['screen_state'][i]: " , memory["screen_state"][i].shape)
        # print("memory['non_space_state'][i] shape : " , memory["non_space_state"][i].shape)
        # print("non_spatial_memory_clip shape: ",non_spatial_memory_clip.shape)
        # print("non_spatial_batch shape: ",non_spatial_batch.shape)
        non_spatial_batch = np.vstack((non_spatial_batch, non_spatial_memory_clip.reshape((*non_spatial_memory_clip.shape)) ))

        next_screen_memory_clip = memory["next_screen_state"][i][j:j+batch_frame_len]
        next_screen_batch = np.vstack((next_screen_batch, next_screen_memory_clip.reshape((*next_screen_memory_clip.shape)) ))
        next_minimap_memory_clip = memory["next_minimap_state"][i][j:j+batch_frame_len]
        next_minimap_batch = np.vstack((next_minimap_batch, next_minimap_memory_clip.reshape(( *next_minimap_memory_clip.shape)) ))
        next_non_spatial_memory_clip = memory["next_non_space_state"][i][j:j+batch_frame_len]
        next_non_spatial_batch = np.vstack((next_non_spatial_batch, next_non_spatial_memory_clip.reshape((*next_non_spatial_memory_clip.shape)) ))

        n_spatial_memory_clip = memory["action_non_spatial"][i][j:j+batch_frame_len]
        non_spatial_action_batch = np.vstack((non_spatial_action_batch, n_spatial_memory_clip.reshape(( *n_spatial_memory_clip.shape)) ))

        primary_spatial_memory_clip = memory["action_spatial_primary"][i][j:j+batch_frame_len]
        primary_spatial_action_batch = np.vstack((primary_spatial_action_batch, primary_spatial_memory_clip.reshape((*primary_spatial_memory_clip.shape)) ))

        #note: we don't seem to be using spatial batches to train, mostly bc we don't have spatial advantage

        reward_memory_clip = memory["reward"][i][j:j+batch_frame_len]
        reward_batch = np.vstack((reward_batch, reward_memory_clip.reshape((*reward_memory_clip.shape)) ))

        value_memory_clip = memory["value"][i][j:j+batch_frame_len]
        current_value_batch = np.vstack((current_value_batch, value_memory_clip)) 
        is_primary_memory_clip = memory["is_primary"][i][j:j+batch_frame_len]
        is_primary_batch =  np.vstack((is_primary_batch, is_primary_memory_clip)) 



    # print("next_screen_batch sahpe : ", next_screen_batch.shape)
    feed_dict = {NN.screen_input_:next_screen_batch, NN.minimap_input_:next_minimap_batch, NN.non_spatial_input_ : next_non_spatial_batch }
    next_value_batch = sess.run( NN.value,  feed_dict = feed_dict )
    advantage_non_spatial_batch = reward_batch + discount_rate*next_value_batch - current_value_batch

    reward_sum = np.zeros(reward_batch[0].shape)
    n=len(screen_batch)
    discount_reward_batch = []
    # print("reward_batch ", reward_batch.shape)
    for i in range(1,n+1):
        reward_sum =reward_batch[-i,:]+ discount_rate*reward_sum
        discount_reward_batch.insert(0, reward_sum)
    discount_reward_batch = np.array(discount_reward_batch)
    # print("reward batch shape",discount_reward_batch.shape)
    # print("advantage_non_spatial_batch shape: ", advantage_non_spatial_batch.shape)
    # print("screen batch shape",screen_batch.shape)
    #backpropgate
    feed_dict={NN.screen_input_: screen_batch,
                                    NN.minimap_input_ : minimap_batch,
                                    NN.non_spatial_input_ : non_spatial_batch,
                                    NN.non_spatial_actions_ : non_spatial_action_batch,
                                    NN.primary_actions_ : primary_spatial_action_batch,
                                    NN.advantages_non_spatial_: advantage_non_spatial_batch,
                                    NN.rewards_: discount_reward_batch,
                                    NN.is_primary: is_primary_batch
                                                                         }

                        
    loss_value, _ =sess.run( [NN.loss_total, NN.apply_grads], #what happens when theres two inputs
                        feed_dict=feed_dict)
    f_loss.write(str(np.mean(loss_value))+" \n")




# def train_iteration(env, sess, agent,exp, saver, total_episodes, dqn_sc2, dqn_target, eps_s, eps_f,decay_rate,time,batch_size , gamma ):
#     time=0
#     time=train(env, sess, agent,exp, saver, total_episodes, dqn_sc2, dqn_target, eps_s, eps_f,decay_rate,time,batch_size , gamma )
#     test(env,  agent, saver,dqn_sc2, exp,  eps_s, eps_f,decay_rate,time)
    
    
#     return time

