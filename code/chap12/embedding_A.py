import numpy as np
from tensorflow.keras.layers import Embedding 
from tensorflow.keras.models import Sequential

#
# Embedding(input_dim,output_dim,input_length)
# - input_dim  : Size of the vocabulary
# - output_dim : Length of the vector for each word
# - input_length : Maximum length of a sequence
#
model = Sequential()
model.add(Embedding(100, 4, input_length=3))
model.summary()
model.compile('rmsprop', 'mse')

# 입력 형태: (batch_size, input_length)=(32, 3)
# 출력 형태: (None, 3, 4)
input_array = np.random.randint(100, size=(32, 3))
input_array.shape
input_array[0] #[0:3]


output_array = model.predict(input_array)
print(output_array.shape)
output_array[0] #[0:3]

#### 3D-Plot ##############################
# https://jehyunlee.github.io/2021/07/10/Python-DS-80-mpl3d2/
###########################################
# from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
##############################

fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(projection='3d')

ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax.set_ylabel("Y", fontdict=fontlabel, labelpad=16)
ax.set_title("Z", fontdict=fontlabel)

for pi in range(len(input_array)):
    x = input_array[pi][0]
    y = input_array[pi][1]
    z = input_array[pi][2]
    # x.shape,y.shape,z.shape
    ax.scatter(x, y, z) #, c=z)
plt.show()    

###############################
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(projection='3d')

ax.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax.set_ylabel("Y", fontdict=fontlabel, labelpad=16)
ax.set_title("Z", fontdict=fontlabel)
# output_array[0][0,:].shape
for pi in range(len(output_array)):
    x = output_array[pi][0,:]
    y = output_array[pi][1,:]
    z = output_array[pi][2,:]
    # x.shape,y.shape,z.shape
    ax.scatter(x, y, z) #, c=z)
plt.show()    

###############################################
fontlabel = {"fontsize":"large", "color":"gray", "fontweight":"bold"}

fig = plt.figure(figsize=(10, 3))
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection="3d")

ax0.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax0.set_ylabel("Y", fontdict=fontlabel, labelpad=16)
ax0.set_title("inputs", fontdict=fontlabel)

for pi in range(len(input_array)):
    x = input_array[pi][0]
    y = input_array[pi][1]
    z = input_array[pi][2]
    ax0.scatter(x, y, z) #, c=z)
    
ax1.set_xlabel("X", fontdict=fontlabel, labelpad=16)
ax1.set_ylabel("Y", fontdict=fontlabel, labelpad=16)
ax1.set_title("outputs", fontdict=fontlabel)

for pi in range(len(output_array)):
    x = output_array[pi][0,:]
    y = output_array[pi][1,:]
    z = output_array[pi][2,:]
    # x.shape,y.shape,z.shape
    ax1.scatter(x, y, z) #, c=z)
plt.show()    