from __future__ import print_function, division
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 5
backprop_length = 4
num_classes = 10
state_size = 8
num_epochs = 10
total_series_length = 5000
echo_step = 3
num_batches = total_series_length//batch_size//backprop_length

#Inputs
batchX = tf.placeholder(tf.float32, [batch_size, backprop_length])
batchY = tf.placeholder(tf.int32, [batch_size, backprop_length])

#Initialize LSTM state
cell_state = tf.placeholder(tf.float32, [batch_size, state_size])
hidden_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state = tf.nn.rnn_cell.LSTMStateTuple(cell_state, hidden_state)

W = tf.Variable(np.random.rand(state_size, num_classes), dtype=tf.float32)
b = tf.Variable(np.random.rand(1, num_classes), dtype=tf.float32)

inputs_series = batchX #tf.split(batchX, backprop_length, 1)   # splits into tensors
labels_series = batchY #tf.unstack(batchY, axis=1)

cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, state_is_tuple=True)
states_series, current_state = tf.nn.static_rnn(cell, inputs_series, init_state)

logits_series = [tf.matmul(state, W) + b for state in states_series]
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)




## TEMP:

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)



with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_cell_state = np.zeros((batch_size, state_size))
        _current_hidden_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * backprop_length
            end_idx = start_idx + backprop_length

            batchXin = x[:,start_idx:end_idx]
            batchYin = y[:,start_idx:end_idx]

            
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX: batchXin,
                    batchY: batchYin,
                    cell_state: _current_cell_state,
                    hidden_state: _current_hidden_state
                })

            print("prediction: ",predictions_series)

            _current_cell_state, _current_hidden_state = _current_state

            loss_list.append(_total_loss)

            if batch_idx%1 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
