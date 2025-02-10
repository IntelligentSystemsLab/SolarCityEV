# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/9/25 14:35
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/9/25 14:35

import torch
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, input_size=1, weekday_size=1,date_size=1, sta_feature=2, hidden_size=3, seq_len=6, output_size=1, num_layers=3):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.rnn_fc = nn.Linear(hidden_size * seq_len, 1)
        self.xconv=nn.Conv1d(in_channels=6,out_channels=16,kernel_size=1)
        self.xlinear=nn.Linear(16, 1)
        self.time1=nn.Linear(weekday_size+date_size,4)
        self.time2=nn.Linear(4,1)
        self.statics1 = nn.Linear(sta_feature, 4)
        self.statics2 = nn.Linear(4, 1)
        self.dense = nn.Linear(4, output_size)
        self.relu=nn.ReLU()

    def forward(self, input_data):
        x = input_data[:, 0:6]
        x = torch.reshape(x, [x.shape[0], x.shape[1], 1])
        x2 = input_data[:, 0:6]
        x2 = torch.reshape(x2, [x.shape[0], x.shape[1], 1])
        sta = input_data[:, 6:8]
        time=input_data[:,8:]
        time[:,0]=time[:,0]/365
        x, _ = self.rnn(x)
        b, s, f = x.shape
        x = x.reshape(b, s * f)
        x = self.rnn_fc(x)
        x2=self.xconv(x2)
        x2=torch.squeeze(x2,-1)
        x2=self.xlinear(x2)
        time=self.time1(time)
        time=self.time2(time)
        sta = self.statics1(sta)
        sta = self.statics2(sta)

        x = torch.cat((x,x2,time, sta), dim=1)
        x = self.dense(x)
        x = self.relu(x)
        return x