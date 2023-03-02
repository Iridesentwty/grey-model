
import numpy as np

class gm_11():
    def __init__(self, data, predict_step=0):
        self.data = data
        self.predict_step = predict_step
        self.sim_values = np.zeros((len(self.data),1))
        self.predict_val = np.zeros((self.predict_step,1))
        self.coff = []
        self.error = []
        self.var_loss = []
    def lsm(self):
        cum_data=np.cumsum(self.data)
        Y = self.data[1:]
        background_values = np.zeros((len(self.data)-1,1))
        for i in range(1, len(self.data)):
            background_values[i-1][0] = 0.5 * cum_data[i]+0.5 * cum_data[i-1]
        one_array = np.ones((len(self.data)-1, 1))
        B = np.hstack((-background_values, one_array))
        self.coff = np.matmul(np.linalg.inv(np.matmul(B.T, B)), np.matmul(B.T, Y))
        return self.coff
    def fit(self):
        self.lsm()
        a = self.coff[0]
        b = self.coff[1]
        self.sim_values[0] = self.data[0]
        for i in range(2,len(self.data)+1):
            self.sim_values[i-1] = (1 - np.exp(a)) * (self.data[0] - b / a) * np.exp(-a * (i - 1))
        self.sim_values = self.sim_values.reshape((1, len(self.data)))[0]
        return self.sim_values
    def predict(self):
        self.lsm()
        a = self.coff[0]
        b = self.coff[1]
        for i in range(len(self.data)+1, len(self.data)+self.predict_step+1):
            self.predict_val[i-len(self.data)-1][0] = (1 - np.exp(a)) * (self.data[0] - b / a) * np.exp(-a * (i - 1))
        self.predict_val = self.predict_val.reshape((1, self.predict_step))
        return self.predict_val
    def errors(self):
        for i in range(len(self.data)):
            self.error.append(self.sim_values[i]-self.data[i])
        return self.error
    def loss(self):
        for i in range(1,len(self.data)):
            self.var_loss.append(np.abs(self.sim_values[i]-self.data[i])/self.data[i])
        return np.average(self.var_loss)
def data(a):
    list = []
    for i in range(7):
        m = np.exp(a*(i))
        list.append(m)
    return list
m = data(0.8)
print(m)
x=[3478, 3955, 4076, 4347, 4390, 4266, 4410, 4706, 5071, 5238, 5377]
model=gm_11(x)
#print(model.lsm())
print("拟合为{0}".format(model.fit()))
#print(model.predict())
print(model.errors())
print(model.loss())














# class Verhulst():#x为原始序列，k为预测长度
#     def __init__(self):
#         self.data=[]
#         self.predict=[]
#     def fit(self,data,predict_steps):
#         self.data=data
#         cum_data=np.cumsum(data)
#         for i in range(2,len(data)+1):
#             x =






