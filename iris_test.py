import numpy as np

import csv
from neuralnet import Neuralnet
class Iris_test:
    def __init__(self,train_data, test_data, lr=0.005, cmax=100000, num_hidden=2, emax= 0.1):
        self.train_data = train_data
        self.test_data = test_data
        self.lr = lr
        self.cmax = cmax
        self.num_hidden = num_hidden
        self.emax = emax
        self.input_test = []
        self.output_test = []
        self.train = []
        self.target = []
    def prepare_data(self):
        for i in self.train_data:
            x=list(i[:-1])
            x= list(np.float_(x))
            self.train.append(x)
        for i in self.test_data:
            x=list(i[:-1])
            x= list(np.float_(x))
            self.input_test.append(x)
        for i in self.train_data:
            x = list(i[-1])
            x = list(np.float_(x))
            self.target.append(x)
        for i in self.test_data:
            x = list(i[-1])
            x = list(np.float_(x))
            x=int(x[0])
            self.output_test.append(x)
        a=0
        for i in self.target:
            if i== [1.0]:
                self.target[a]=[1.0, .0, .0]
            if i == [2.0]:
                self.target[a] = [.0, 1.0, .0]
            if i == [3.0]:
                self.target[a] = [.0, .0, 1.0]
            a+=1
        self.train=np.array(self.train)
        self.target=np.array(self.target)
    def train_model(self):
        self.prepare_data()
        self.model= Neuralnet(self.train,self.target,num_input=4,num_hidden=self.num_hidden,num_output=3, lr=self.lr, cmax=self.cmax,emax=self.emax)
        self.model.train()
    def test_model(self):
        a = self.model.forward(self.input_test)
        self.NN_outputs = []
        for i in a:
            max_value = max(i)
            max_index = list(i).index(max_value)
            self.NN_outputs.append(max_index+1)
        total = 0
        correct = 0
        for i in range(len(self.output_test)):
            if self.output_test[i] == self.NN_outputs[i]:
                correct += 1
            total +=1
        return(correct/total)


