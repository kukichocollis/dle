# import numpy as np
import tensorflow as tf
inputs = tf.random.normal([32, 10, 8])
print(inputs.shape)

lstm = tf.keras.layers.LSTM(4)		# 4개의 셀을 가진다. 
output = lstm(inputs)

print(output.shape)
print(output)

###############################
# RNN
#######
rnn = tf.keras.layers.SimpleRNN(4)		# 4개의 셀을 가진다. 
output2 = rnn(inputs)

print(output2.shape)
print(output2)
