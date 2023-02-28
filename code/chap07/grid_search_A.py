# import numpy as np 
# import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.model_selection import GridSearchCV
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
# OS warning cure
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 데이터 세트 준비
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)
                             
# 신경망 모델 구축
def build_model():
    network = tf.keras.models.Sequential()
    network.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(tf.keras.layers.Dense(10, activation='sigmoid'))

    network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return network

# 하이퍼 매개변수 딕셔너리
param_grid = {
              'epochs':[1, 2, 3],	# 에포크 수: 1, 2, 3
              'batch_size':[32, 64, 100]	# 배치 크기: 32, 64, 100
             }

# 케라스 모델을 scikeras에서 사용하도록 포장한다. 
model = KerasClassifier(model = build_model, verbose=1)

## Elapsed time start
import time
start = time.time()

# 그리드 검색
gs = GridSearchCV(
    estimator=model,
    param_grid=param_grid, 
    cv=3, 
    # n_jobs=-1 # comment this line to avoid out-of-memory
)

# 그리드 검색 결과 출력
grid_result = gs.fit(train_images, train_labels)

## Elapsed time end
end = time.time()
print('Elapsed time:', end - start) # Elapsed time: 122.5, 189.5

print(grid_result.best_score_) # 0.9725, 0.9711
print(grid_result.best_params_) # {'batch_size': 64, 'epochs': 3}
