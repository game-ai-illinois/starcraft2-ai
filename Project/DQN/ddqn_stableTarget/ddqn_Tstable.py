###############################################################################

#
#   File Created: Thurs 2018
#   Author: Hyeon-Seo Yun
#   File: dqn.py
#   Path: 
#   References: 
#       https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8
#   
#

###############################################################################

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
import numpy as np
import tensorflow as tf
from collections import deque
from modified_state_space import state_modifier

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

def get_reward( obs, done):
    my_units = get_units_by_type(obs,units.Terran.Marine)
    zergling  = get_units_by_type(obs,units.Zerg.Zergling)
    baneling  = get_units_by_type(obs,units.Zerg.Baneling)
    flag =True
    if not done:
        return  0.5*len(my_units) - len(zergling) -len(baneling) -10 #extra minus 10 for living
        # return len(my_units) - len(zergling) -len(baneling) -10 #extra minus 10 for living
    else:
        if len(zergling) +len(baneling) ==0: #if you won the game
            return 0.5*len(my_units) - len(zergling) -len(baneling) -10 + 1000 #add big reward for winning
            # return len(my_units) - len(zergling) -len(baneling) -10 + 1000 #add big reward for winning
        else:
            return 0.5*len(my_units) - len(zergling) -len(baneling) -10 - 1000 #add big punishment for losing
            # return len(my_units) - len(zergling) -len(baneling) -10 - 1000 #add big punishment for losing

def transform_state(timesteps):
    state =state_modifier.modified_state_space(timesteps[0])[0]
    print("feature screen: ",timesteps[0].observation.feature_screen.shape)
    print("feature mini map: ",timesteps[0].observation.feature_minimap.shape)
    print("non-spatial features: ",timesteps[0].observation.player)
    # print("state : ", state)
    # state= (state[0].flatten())
    # state=state.reshape(1, state.shape[0])
    # print(state.shape)
    return state

def trans_action(action_arg , array_size =32+40+1): # action is assumed to be the arg value of the action layer
    return_array = np.zeros(array_size)
    return_array[action_arg] =1
    return return_array

def argmax_action(arg , timestep_obs):
    action_array = trans_action(arg)
    action =actions.FUNCTIONS.no_op()

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
    return action, action_array

def eps_decay(obs, sess, DQN_sc2, epsilon, timestep_obs):
    action =actions.FUNCTIONS.no_op()
    arg=None #default value for debugging

    
    if np.random.random(1) < epsilon:
        #explore
        arg = np.random.randint(0,32+40+1)
           
    else:
        #exploit
        Qs = sess.run(DQN_sc2.Yhat, feed_dict = {DQN_sc2.inputs_: obs})    
        
        # Take the biggest Q value (= the best action)
        arg = np.argmax(Qs)
        print(arg)
    
    action_array = trans_action(arg)
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
    return action, action_array
        
        
#     print("action shape: ", action.shape)
    return action , action_array


class experience:
    def __init__(self, max_size):
        self.buffer = deque(maxlen = max_size)
        
    def add(self, exp_set):
        self.buffer.append(exp_set)
    
    def sample(self,size):
        buffer_size=len(self.buffer)
        #print(buffer_size)
        #print(size)
        indeces=np.random.choice(buffer_size, size , replace=False)
        
        return [self.buffer[i] for i in indeces ]

class State:
    def __init__(self, max_size):
        self.transformed_obs = deque(maxlen = max_size)
        self.size =max_size
        
    def add(self, transformed_obs):
        self.transformed_obs.append(transformed_obs)

    def reinitialize(self, transformed_obs):
        for i in range(self.size ):
            self.transformed_obs.append(transformed_obs)    
    
    def get(self):
        # print("return shape", np.array(self.transformed_obs).reshape((84, 84,self.size *7 )).shape)
        return  np.array(self.transformed_obs).reshape((84, 84,self.size *7 ))

class DQN:
    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name
        lr = self.learning_rate
        D=(self.state_size)
        K=(self.action_size)
        M1=500
        M2=500
        M3=500
        # print("D: ",D)
        #variable scope function from https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
        with tf.variable_scope(self.name):
            self.inputs_= tf.placeholder(tf.float32, shape=[None, *D], name="inputs")
            self.actions_= tf.placeholder(tf.float32, shape=[None, K], name='actions')
            self.target_Q = tf.placeholder(tf.float32, [None,], name="target")

            #convolution layer from doom code
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                            filters = 32,
                                            kernel_size = [8,8],
                                            strides = [4,4],
                                            padding = "VALID",
                                            kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                            name = "conv1")
            
            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'batch_norm1')
            
            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [20, 20, 32]
            
            
            self.conv2 = tf.layers.conv2d(inputs = self.conv1_out,
                                    filters = 64,
                                    kernel_size = [4,4],
                                    strides = [2,2],
                                    padding = "VALID",
                                    kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    name = "conv2")
        
            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [9, 9, 64]
            

            self.conv3 = tf.layers.conv2d(inputs = self.conv2_out,
                                    filters = 128,
                                    kernel_size = [4,4],
                                    strides = [2,2],
                                    padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                    name = "conv3")
        
            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                    training = True,
                                                    epsilon = 1e-5,
                                                        name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [3, 3, 128]
            
            
            self.flatten = tf.layers.flatten(self.conv3_out)
            ## --> [1152]

            self.hiddenlayer = tf.layers.dense( self.flatten , units=500, activation = tf.nn.elu)
            


            self.W1=tf.Variable(tf.random_normal([500,M1], stddev=0.1))
            self.b1=tf.Variable(tf.random_normal([M1], stddev=0.1))
            self.W2=tf.Variable(tf.random_normal([M1,M2], stddev=0.1))
            self.b2=tf.Variable(tf.random_normal([M2], stddev=0.1))
            self.W3=tf.Variable(tf.random_normal([M2,M3], stddev=0.1))
            self.b3=tf.Variable(tf.random_normal([M3], stddev=0.1))
            # self.W4=tf.Variable(tf.random_normal([M3,K], stddev=0.1))
            # self.b4=tf.Variable(tf.random_normal([K], stddev=0.1))
    #initialization of weights and biases

            self.Z1 = tf.nn.relu( tf.matmul(self.hiddenlayer, self.W1) + self.b1 )
            self.Z2 = tf.nn.relu( tf.matmul( self.Z1, self.W2) + self.b2 )
            self.Z3 = tf.nn.relu( tf.matmul(self.Z2, self.W3) + self.b3 )

            #vlaue and advantage layers and fcs are from  https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
            self.value_fc = tf.layers.dense(inputs = self.Z3,
                                  units = 512, #512 output nodes
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value_fc")
            
            self.value = tf.layers.dense(inputs = self.value_fc,
                                        units = 1,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="value")
            
            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs = self.Z3,
                                  units = 512,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantage_fc")
            
            self.advantage = tf.layers.dense(inputs = self.advantage_fc,
                                        units = self.action_size,
                                        activation = None,
                                        kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="advantages")
            
            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.Yhat = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
              
            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.Yhat, self.actions_), axis=1)
            self.cost = tf.reduce_sum(tf.square(self.target_Q - self.Q))
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost) #tf.train.AdamOptimizer(lr).minimize(self.cost)


 
#update target graph implementation from https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682
def update_target_graph():
    
    # Get the parameters of our dqn_sc2
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn_sc2")
    
    # Get the parameters of our dqn_target
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "dqn_target")

    
    # Update our target_network parameters with DQNNetwork parameters
    for from_var,to_var in zip(from_vars,to_vars):
        to_var.assign(from_var)
    

def train(env, sess, agent,exp, saver, total_episodes,dqn_sc2, dqn_target, eps_s, eps_f,decay_rate,time,batch_size , gamma  ):
    
    
    state = State(4) #initialize state
    for episode in range(total_episodes):
        
        print("training")
        iteration_num = 10
        for training_iteration in range(iteration_num):
            # Make a new episode and observe the first state
            timesteps = env.reset()
            state_frame = transform_state(timesteps)
            state.reinitialize(state_frame) #initilize state variable

            while True:
                state_old = state
                if eps_s - decay_rate*time < eps_f:
                    epsilon= eps_f
                else:
                    epsilon =eps_s - decay_rate*time
                [step_actions , step_actions_array] = eps_decay(state.get().reshape((1,*(state.get().shape) )),sess,dqn_sc2, epsilon ,timesteps[0] )
                # print("eps action: ", step_actions)
                done = False
                if timesteps[0].last():
                    done=True
                reward = get_reward(timesteps[0],done)
                timesteps = env.step([step_actions])
                state.add( transform_state(timesteps) )
                
                exp.add([state_old.get(),step_actions_array,  reward, state.get(), done])
                if timesteps[0].last():
                    break
                time+=1 #increase time for eps_decay
            
  
        print("learning!")
        
        batch = exp.sample(batch_size)
        states_mb=[]
        actions_mb=[]
        rewards_mb=[]
        next_states_mb=[]
        dones_mb=[]
        for i in batch:
            states_mb.append(i[0])
            actions_mb.append(i[1] )
            rewards_mb.append(i[2])
            next_states_mb.append(i[3])
            dones_mb.append(i[4])
        states_mb=np.array(states_mb)
        states_mb=np.squeeze(states_mb)
        actions_mb=np.array(actions_mb)
        actions_mb=np.atleast_2d(actions_mb)
        rewards_mb=np.array(rewards_mb)
        next_states_mb=np.array(next_states_mb)
        next_states_mb=np.squeeze(next_states_mb)
        dones_mb=np.array(dones_mb)
        
        target_Qs_batch = []

        # Get Q values for next_state from target network
        Qs_next_state = sess.run(dqn_target.Yhat, feed_dict = {dqn_target.inputs_: next_states_mb})

        # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
        for i in range(0, batch_size):
            terminal = dones_mb[i]
            # If we are in a terminal state, only equals reward
            if terminal:
                target_Qs_batch.append(rewards_mb[i])
            else:
                target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                target_Qs_batch.append(target)

        targets_mb = np.array([each for each in target_Qs_batch])
        
        #backpropgate
        sess.run( dqn_sc2.optimizer, #what happens when theres two inputs
                            feed_dict={dqn_sc2.inputs_: states_mb,
                                        dqn_sc2.target_Q: targets_mb,
                                        dqn_sc2.actions_: actions_mb})
    
        # Save model every 5 episodes
        if episode % 10 == 0:
            save_path = saver.save(sess, "./models/dqn_split.ckpt")
            #check if the weights are different
            variable_current = sess.run(dqn_sc2.W3)
            variable_target = sess.run(dqn_target.W3)
            print("weights same?: ", np.array_equal(variable_current,variable_target))
            
            #update target network as the current one
            update_target_graph()
            print("Model Saved")
    return time


def test(env,  agent, saver,dqn_sc2, exp,  eps_s, eps_f,decay_rate,time):
    print("testing!")
    agent.setup(env.observation_spec(), env.action_spec())
    # Load the model
    with tf.Session() as sess:
        saver.restore(sess, "./models/dqn_split.ckpt")
        state = State(4)
        for i in range(3):
            agent.reset()
            timesteps = env.reset()
            
            state.reinitialize(transform_state(timesteps) )
            totalScore = 0
            done=False
            while not done:
                # Take the biggest Q value (= the best action)
            
                Qs = sess.run(dqn_sc2.Yhat, feed_dict = {dqn_sc2.inputs_: state.get().reshape((1,*(state.get().shape) ))}) #run NN
                print(np.argmax(Qs))
                if eps_s - decay_rate*time < eps_f:
                    epsilon= eps_f
                else:
                    epsilon =eps_s - decay_rate*time
                [step_actions , step_actions_array] = eps_decay(state.get().reshape((1,*(state.get().shape) )),sess,dqn_sc2, epsilon ,timesteps[0] )
                state_old = state
                timesteps = env.step([step_actions])
                state.add(transform_state(timesteps))
                
                
                if timesteps[0].last():
                    done=True
                reward = get_reward(timesteps[0],done)
                
                exp.add([state_old.get(),step_actions_array,  reward, state.get(), done])

                totalScore += reward

            print("TOTAL_SCORE", totalScore)

def train_iteration(env, sess, agent,exp, saver, total_episodes, dqn_sc2, dqn_target, eps_s, eps_f,decay_rate,time,batch_size , gamma ):
    time=0
    time=train(env, sess, agent,exp, saver, total_episodes, dqn_sc2, dqn_target, eps_s, eps_f,decay_rate,time,batch_size , gamma )
    test(env,  agent, saver,dqn_sc2, exp,  eps_s, eps_f,decay_rate,time)
    
    
    return time

