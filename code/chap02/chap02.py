# cap02 - python basics
#
3.14 * 10 * 10 

3.14 * 10**2


type(10)

type(3.14)

type("python")


r = 20

PI = 3.14		# 원주율 정의

area = PI * r**2

area

# list
lst = [ 10, 20, 30, 40, 50]		# 리스트 정의

lst

# indexing & slicing
lst[2] 				# 리스트의 요소 접근

lst[2] = 90			# 세 번째 요소를 90으로 변경

lst

len(lst)

lst[0:3]

lst[2:]	

lst[:3]

lst[:-1]

# dictionary
car = { 'HP':200, 'make': "BNW" }	# 딕셔너리 정의

car['HP']

car['color'] = "white"

car

# Conditional
temp = -10

if temp < 0 :
    print("영하입니다.")	
else:
    print("영상입니다.")

# loop
for i in [1, 2, 3, 4, 5] :
    print(i, end=" ")

# functio
def sayHello():
    print("Hello!")

sayHello()

# class
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def sayHello(self):
    print("Hello 나의 이름은 " + self.name)

p1 = Person("John", 36)
p1.sayHello()

# [DIY] 이름과 나이를 다 출력하게 코드를 변경하시오.
#################################################
