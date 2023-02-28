# chap02 - numpy
import numpy as np

# ndarray
a = np.array([1, 2, 3])

a

a[0]

b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

b

b[0][2]

a = np.array([[ 0, 1, 2],
 		[ 3, 4, 5],
		[ 6, 7, 8]])

a.shape

a.ndim

a.dtype

a.itemsize

a.size


np.zeros( (3, 4) )	

np.ones( (3, 4), dtype=np.int32 )

np.eye(3)

x = np.ones(5, dtype=np.int64)

x

np.arange(5)

np.arange(1, 6)

np.arange(1, 10, 2)

np.linspace(0, 10, 100)

# array
arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])

np.sort(arr)

# 배열 합치기
x = np.array([[1, 2], [3, 4]])

y = np.array([[5, 6], [7, 8]])

np.concatenate((x, y), axis=1)

np.vstack((x, y))

np.hstack((x, y))

# reshape
a = np.arange(12)

a.shape

a.reshape(3, 4)

a.reshape(6, -1)

array = np.arange(30).reshape(-1, 10)

array 

arr1, arr2 = np.split(array, [3], axis=1)

arr1

arr2

# 차원 확장
a = np.array([1, 2, 3, 4, 5, 6])

a.shape

a1 = a[np.newaxis, :]

a1

a1.shape

a2 = a[:, np.newaxis]

a2.shape

# indexing & slicing
ages = np.array([18, 19, 25, 30, 28])

ages[1:3] 

ages[:2] 

y = ages > 20

y

ages[ ages > 20 ]

# 인덱싱
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a[0, 2]

a[0, 0] = 12

a

# 슬라이싱
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a[0:2, 1:3]

# operation of arrays
arr1 = np.array([[1, 2], [3, 4], [5, 6]])

arr2 = np.array([[1, 1], [1, 1], [1, 1]])

result = arr1 + arr2

result

miles = np.array([1, 2, 3])

result = miles * 1.6

result

arr1 = np.array([[1, 2], [3, 4], [5, 6]])

arr2 = np.array([[2, 2], [2, 2], [2, 2]])

result = arr1 * arr2

result

arr1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

arr2 = np.array([[2, 2], [2, 2], [2, 2]])

result = arr1 @ arr2

result

A = np.array([0, 1, 2, 3])

10 * np.sin(A)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

a.sum()

a.min()

a.max()

scores = np.array([[99, 93, 60], [98, 82, 93],
               [93, 65, 81], [78, 82, 81]])    

scores.mean(axis=0)

# random numbers
np.random.seed(100)

np.random.rand(5)

np.random.rand(5, 3)

np.random.randn(5)

np.random.randn(5, 4)

m, sigma = 10, 2

m + sigma*np.random.randn(5)

mu, sigma = 0, 0.1 	# 평균과 표준 편차

np.random.normal(mu, sigma, 5)

# unique()
a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

unique_values = np.unique(a)

unique_values

# Transpose of an array
arr = np.array([[1, 2], [3, 4], [5, 6]])
print(arr.T)

x = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])


x.flatten()

########################
# Pandas
import numpy as np
import pandas as pd

x = pd.read_csv('countries.csv', header=0).values
print(x)

########################
# matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

X = [ "Mon", "Tue", "Wed", "Thur", "Fri",  "Sat", "Sun" ] 
Y1 = [15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
Y2 = [20.1, 23.1, 23.8, 25.9, 23.4, 25.1, 26.3]

plt.plot(X, Y1, label="Seoul")		# 분리시켜서 그려도 됨
plt.plot(X, Y2, label="Busan")		# 분리시켜서 그려도 됨
plt.xlabel("day")
plt.ylabel("temperature")
plt.legend(loc="upper left")
plt.title("Temperatures of Cities")
plt.show()


import matplotlib.pyplot as plt
# %matplotlib inline

X = [ "Mon", "Tue", "Wed", "Thur", "Fri",  "Sat", "Sun" ] 
plt.plot(X, [15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4], "sm")
plt.show()


import matplotlib.pyplot as plt
# %matplotlib inline

X = [ "Mon", "Tue", "Wed", "Thur", "Fri",  "Sat", "Sun" ] 
Y = [15.6, 14.2, 16.3, 18.2, 17.1, 20.2, 22.4]
plt.bar(X, Y)
plt.show()



import matplotlib.pyplot as plt
import numpy as np

numbers = np.random.normal(size=10000)

plt.hist(numbers)
plt.xlabel("value")
plt.ylabel("freq")
plt.show()


import matplotlib.pyplot as plt
import numpy as np
X = np.arange(0, 10)
Y = X**2
plt.plot(X, Y)
plt.show()


X = np.arange(0, 10)
Y1 = np.ones(10)
Y2 = X 
Y3 = X**2 
plt.plot(X, Y1, X, Y2, X, Y3)
plt.show()


import matplotlib.pyplot as plt 
import numpy as np 
  
X = np.linspace(-10, 10, 101) 
Y = 1/(1 + np.exp(-X)) 
  
plt.plot(X, Y) 
plt.xlabel("x") 
plt.ylabel("Sigmoid(X)") 
plt.show() 


import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)  		# 시그모이드 함수 1차 미분 함수
    return s,ds

X = np.linspace(-10, 10, 101) 
Y1, Y2 = sigmoid(X)
  
plt.plot(X, Y1, X, Y2) 
plt.xlabel("x") 
plt.ylabel("Sigmoid(X), Sigmoid'(X)") 
plt.show() 





