import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# 2d-list => 2-d array
def make_sample(data, window):
    train = []					# 공백 리스트 생성
    target = []
    for i in range(len(data)-window):		# 데이터의 길이만큼 반복
        train.append(data[i:i+window])		# i부터 (i+window-1) 까지를 저장
        target.append(data[i+window])		# (i+window) 번째 요소는 정답
    return np.array(train), np.array(target)	# 파이썬 리스트를 넘파이로 변환

seq_data = []
for i in np.arange(0, 1000):
    seq_data += [[np.sin( np.pi * i* 0.01 )]]
X, y = make_sample(seq_data, 10)		# 윈도우 크기=10

print(X.shape,y.shape) 

# RNN model
model = Sequential()
model.add(SimpleRNN(10, activation='tanh', input_shape=(10,1)))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=100, verbose=1)

plt.plot(history.history['loss'], label="loss")
plt.show()

# Test data
seq_data = []
for i in np.arange(0, 1000):			# 테스트 샘플 생성
    seq_data += [[np.cos( np.pi * i* 0.01 )]]
X2, y2 = make_sample(seq_data, 10)		# 윈도우 크기=10
X2.shape,y2.shape

# Prediction
y_pred = model.predict(X2, verbose=0)		# 테스트 예측값
plt.plot(np.pi * np.arange(0, 990)*0.01, y_pred, label='pred' )
plt.plot(np.pi * np.arange(0, 990)*0.01, y2, label='orig')
plt.legend()
plt.show()