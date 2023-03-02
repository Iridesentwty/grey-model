# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sko.DE import DE

class MPGM_model():
    def __init__(self, data, pre_step=0):
        self.data = data
        self.time_function_up = []
        self.time_function_down = []
        self.pre_step = pre_step
        self.sim_values = np.zeros((len(self.data)+self.pre_step, 1))
        self.history = []
        self.mape = []
        self.errors = []

    def lsm(self, alpha = 0.5):
        cum_data=np.cumsum(self.data)
        Y = self.data[1:]
        background_values = np.zeros((len(self.data)-1, 1))
        for i in range(1, len(self.data)):
            background_values[i-1][0] = (1-alpha) * cum_data[i] + alpha * cum_data[i-1]
        one_array = np.ones((len(self.data)-1, 1))
        B = np.hstack((-background_values, one_array))
        coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
        return coff

    def time_resp_func(self, a, b):
        values = np.zeros((len(self.data)+self.pre_step,1))
        values[0] = self.data[0]
        for i in range(2, len(self.data)+1+self.pre_step):
            values[i-1] = (1 - np.exp(a)) * (self.data[0] - b / a) * np.exp(-a * (i - 1))
        values = values.reshape((1, len(self.data) + self.pre_step))[0]
        return values

    def paramter(self, belta):
        belta1, belta2 = belta
        error_list = []
        ratio_list = []
        for i in range(2, len(self.data)):
            ratio = self.data[i]/self.data[i-1]
            ratio_list.append(ratio)
        w = (np.max(ratio_list)-np.min(ratio_list))/np.average(ratio_list)
        alpha1 = 0.5 + w
        alpha2 = 0.5 - w
        a1, b1 = self.lsm(alpha1)
        a2, b2 = self.lsm(alpha2)
        self.time_function_up = self.time_resp_func(a1, b1)
        self.time_function_down = self.time_resp_func(a2, b2)
        for i in range(1, len(self.data)):
            error = np.abs(self.data[i] - belta1 * self.time_function_down[i] - belta2 * self.time_function_up[i])
            error_list.append(error)
        f = np.sum(error_list)
        return f

    def fit(self):
        de = DE(func=self.paramter, n_dim=2, lb=[0,0], ub=[1,1], max_iter=200)
        best_x, best_y = de.run()
        self.history = de.generation_best_Y
        self.sim_values[0] = self.data[0]
        for i in range(1, len(self.data)+self.pre_step):
            self.sim_values[i] = best_x[0]*self.time_function_down[i] + best_x[1]*self.time_function_up[i]
        self.sim_values = self.sim_values.reshape((1, len(self.data)+self.pre_step))[0]
        return self.sim_values

    def plot(self):
        data = pd.DataFrame(self.history).values.flatten()
        plt.figure(figsize=(15, 8), dpi=140)
        plt.tick_params(labelsize=15)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.xlabel("Epochs", fontsize=18)
        plt.ylabel("TAE", fontsize=18)
        plt.plot(range(0, 200, 1), data)
        plt.yticks(np.arange(min(data), max(data), step=50))
        plt.grid(axis='y', linestyle="--", alpha=1)
        plt.show()

    def get_mape(self):
        for i in range(1, len(self.data)):
            mape = np.abs(self.sim_values[i]-self.data[i])/self.data[i]
            self.mape.append(mape)
        return np.average(self.mape)

    def get_errors(self):
        for i in range(1, len(self.data)):
            errors = np.abs(self.sim_values[i]-self.data[i])
            self.errors.append(errors)
        return np.average(self.errors)

def data(a):
    list = []
    for i in range(7):
        m = np.exp(a*(i))
        list.append(m)
    return list

# m = data(0.5)
# print(m)
x=[3478, 3955, 4076, 4347, 4390, 4266, 4410, 4706, 5071, 5238]
model = MPGM_model(x, pre_step= 3)
print('拟合值为:', model.fit())
print('数据的mape为:', model.get_mape())
model.plot()

