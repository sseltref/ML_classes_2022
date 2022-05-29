import numpy as np

import csv
from neuralnet import Neuralnet

with open('iris.train.csv', newline='') as f:
    reader = csv.reader(f)
    train_data = list(reader)
with open('iris.test.csv', newline='') as f:
    reader = csv.reader(f)
    test_data = list(reader)
input_test =[]
output_test = []
train=[]
target=[]
for i in train_data:
    x=list(i[:-1])
    x= list(np.float_(x))
    train.append(x)
for i in test_data:
    x=list(i[:-1])
    x= list(np.float_(x))
    input_test.append(x)
for i in train_data:
    x = list(i[-1])
    x = list(np.float_(x))
    target.append(x)
for i in test_data:
    x = list(i[-1])
    x = list(np.float_(x))
    x=int(x[0])-1
    output_test.append(x)
a=0
for i in target:
    if i== [1.0]:
        target[a]=[1.0, .0, .0]
    if i == [2.0]:
        target[a] = [.0, 1.0, .0]
    if i == [3.0]:
        target[a] = [.0, .0, 1.0]
    a+=1
train=np.array(train)
target=np.array(target)


model= Neuralnet(train,target,num_input=4,num_hidden=5,num_output=3)
model.train()
a = model.forward(input_test)
NN_outputs = []
for i in a:
    max_value = max(i)
    max_index = list(i).index(max_value)
    NN_outputs.append(max_index)
print(output_test)
print(NN_outputs)
total = 0
correct = 0
for i in range(len(output_test)):
    if output_test[i] == NN_outputs[i]:
        correct += 1
    total +=1
print('Accuracy: ', correct/total)





