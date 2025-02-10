# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023/9/25 15:47
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023/9/25 15:47
import copy
import datetime
import os
import time
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
# import pmdarima as pm
from model.baselines import LSTMv1, FCNN, FGN
from utils import CreateDataset, divide_dataset, calculate_metrics
from model.lstm import MyLSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def day_to_date(year, day):
    fir_day = datetime.datetime(year, 1, 1)
    zone = datetime.timedelta(days=day - 1)
    return datetime.datetime.strftime(fir_day + zone, "%Y-%m-%d")

def meta_train(data,evaluation_data, batch_size, epochs,divide_mode,support_epochs,custom_epochs,log_file,mode, lr=0.005, city_name='',net_path='/results/model/pt_files/Meta_Net_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) +'.pt',print_details=False,save_pre_result=True):
    station_id = list(data.keys())
    evaluation_station_id = list(evaluation_data.keys())
    net = MyLSTM().to(device)
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    if save_pre_result:
        all_id=station_id+evaluation_station_id
        start_date = datetime.date(2023,1,1)
        end_date = datetime.date(2023, 12, 31)
        temp_date = start_date
        data_column=['id','longitude','latitude']
        for i in range((end_date - start_date).days + 1):
            data_column.append(str(temp_date))
            temp_date += datetime.timedelta(days=1)
        pre_result_df=pd.DataFrame(columns=data_column)
        for s in station_id:
            row_data_column = ['s'+str(int(s))+'-ori', data[s][2][0][0], data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
            row_data_column = ['s'+str(int(s))+'-pre', data[s][2][0][0], data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        for s in evaluation_station_id:
            row_data_column = ['s'+str(int(s))+'-ori', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
            row_data_column = ['s'+str(int(s))+'-pre', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    for p in tqdm(range(epochs), desc='Training'):
        net_dict = dict()
        gradient_dict = dict()
        for name, param in net.named_parameters():
            if param.requires_grad:
                gradient_dict[name] = torch.zeros(param.shape).to(device)
        for station in station_id:
            charging_sample = data[station][0]
            charging_lable = data[station][1]
            feature = data[station][2]
            date = data[station][3]
            weekday = data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday,divide_mode, divide_rate=0.8)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            net_dict[station] = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            # net_dict[f] = MyNets.MyLSTM_no_sample()  # no sample
            temp_optimizer = torch.optim.Adam(net_dict[station].parameters(), lr=lr, weight_decay=1e-5)

            # plt.ion()
            for e in range(support_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday),dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = net_dict[station](input_data)
                    # output = net_dict[f](in_feature, nearby_Gaussian_covid)  # without sample
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()

            metrics_matrix=[0,0,0,0,0,0]
            num=0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num+=label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday),dim=1).to(device)
                temp_optimizer.zero_grad()
                output = net_dict[station](input_data)
                loss = temp_loss_function(output, label)
                loss.backward()
                temp_optimizer.step()
                RMSE, MAE, MAPE, MedAE, R2,  EVS = calculate_metrics(output.cpu().detach().numpy()*1, label.cpu().detach().numpy()*1)
                metrics_matrix[0] += RMSE * label.shape[0]
                metrics_matrix[1] += MAE * label.shape[0]
                metrics_matrix[2] += MAPE*label.shape[0]
                metrics_matrix[3] += MedAE * label.shape[0]
                metrics_matrix[4] += R2 * label.shape[0]
                metrics_matrix[5] += EVS * label.shape[0]
            for i in range(6):
                metrics_matrix[i]=metrics_matrix[i]/num
            if print_details:
                print('TRAIN on ID '+str(station)+': RMSE, MAE, MAPE, MedAE, R2,  EVS')
                print(metrics_matrix)

            for name, param in net_dict[station].named_parameters():
                if param.requires_grad:
                    gradient = copy.deepcopy(param.grad.data)
                    gradient_dict[name] += gradient

        for name, param in net.named_parameters():
            if param.requires_grad:
                param.grad = gradient_dict[name] / len(station_id)
        optimizer.step()
        optimizer.zero_grad()
        if print_details:
            print('--------------------------------------------------------------------------')
        total_matrix = [0, 0, 0, 0, 0,0]
        for station in evaluation_station_id:
            charging_sample = evaluation_data[station][0]
            charging_lable = evaluation_data[station][1]
            feature = evaluation_data[station][2]
            date = evaluation_data[station][3]
            weekday = evaluation_data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday, divide_mode,divide_rate=0.8,devide=False)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            temp_net = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            temp_optimizer = torch.optim.Adam(temp_net.parameters(), lr=lr, weight_decay=1e-5)

            for e in range(custom_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = temp_net(input_data)
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()
            metrics_matrix = [0, 0, 0, 0,0, 0]
            num = 0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num += label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                output = temp_net(input_data)
                RMSE, MAE, MAPE,MedAE, R2, EVS = calculate_metrics(output.cpu().detach().numpy()*1, label.cpu().detach().numpy()*1)
                metrics_matrix[0] += RMSE * label.shape[0]
                metrics_matrix[1] += MAE * label.shape[0]
                metrics_matrix[2] += MAPE * label.shape[0]
                metrics_matrix[3] += MedAE * label.shape[0]
                metrics_matrix[4] += R2 * label.shape[0]
                metrics_matrix[5] += EVS * label.shape[0]
                if p==epochs-1 and save_pre_result:
                    date_numpy=date.cpu().detach().numpy()
                    output_numpy=output.cpu().detach().numpy()
                    label_numpy=label.cpu().detach().numpy()
                    num_date=date_numpy.shape[0]
                    for l in range(num_date):
                        day=str(day_to_date(2023,date_numpy[l][0]))
                        pre_result_df.loc[pre_result_df['id'] == 's'+str(int(station))+'-ori', day] = label_numpy[l]
                        pre_result_df.loc[pre_result_df['id'] == 's'+str(int(station))+'-pre', day] = output_numpy[l]
            for i in range(6):
                metrics_matrix[i] = metrics_matrix[i] / num
                total_matrix[i]+=metrics_matrix[i]/len(evaluation_station_id)
            if print_details:
                print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
                print(metrics_matrix)
        if print_details:
            print('--------------------------------------------------------------------------')
        print('\n'+str(p)+'/'+str(epochs)+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS')
        print(total_matrix)
        log_file.writelines(
            str(p)+'/'+str(epochs)+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS'+ '\n'+ str(total_matrix)+'\n'
        )
        log_file.flush()
        if print_details:
            print('##########################################################################')
    if save_pre_result:
        for station in station_id:
            charging_sample = data[station][0]
            charging_lable = data[station][1]
            feature = data[station][2]
            date = data[station][3]
            weekday = data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday,divide_mode, divide_rate=0.8,devide=False)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            temp_net = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            temp_optimizer = torch.optim.Adam(temp_net.parameters(), lr=lr, weight_decay=1e-5)

            for e in range(custom_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = temp_net(input_data)
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()
            num = 0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num += label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                output = temp_net(input_data)
                date_numpy=date.cpu().detach().numpy()
                output_numpy=output.cpu().detach().numpy()
                label_numpy=label.cpu().detach().numpy()
                num_date=date_numpy.shape[0]
                for l in range(num_date):
                    day=str(day_to_date(2023,date_numpy[l][0]))
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day] = label_numpy[l]
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day] = output_numpy[l]
        
        if not os.path.exists('/results/data/pre_result_'+divide_mode+ '/'):
            os.makedirs('/results/data/pre_result_'+divide_mode+ '/')
        pre_result_df.to_csv('/results/data/pre_result_'+divide_mode+ '/' + city_name+'_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.csv', index=False)
    # torch.save(net.state_dict(),'model/pt_files/Meta_Net_'+city_name+'_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) +'.pt')


def meta_train_baselines(data,evaluation_data, batch_size, epochs,divide_mode,support_epochs,custom_epochs,log_file,mode, lr=0.005, city_name='',net_path='model/pt_files/Meta_Net_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) +'.pt',print_details=False,save_pre_result=True,baseline_name=''):
    station_id = list(data.keys())
    evaluation_station_id = list(evaluation_data.keys())
    if baseline_name=='':
        net = MyLSTM().to(device)
    elif baseline_name[:4]=='ours':
        net= MyLSTM().to(device)
    elif baseline_name[:4]=='FCNN':
        net= FCNN().to(device)
    elif baseline_name[:6]=='LSTMv1':
        net= LSTMv1().to(device)
    elif baseline_name[:3]=='FGN':
        net= FGN().to(device)
    loss_function = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    if save_pre_result:
        all_id=station_id+evaluation_station_id
        start_date = datetime.date(2023,1,1)
        end_date = datetime.date(2023, 12, 31)
        temp_date = start_date
        data_column=['id','longitude','latitude']
        for i in range((end_date - start_date).days + 1):
            data_column.append(str(temp_date))
            temp_date += datetime.timedelta(days=1)
        pre_result_df=pd.DataFrame(columns=data_column)
        for s in station_id:
            row_data_column = ['s'+str(int(s))+'-ori', data[s][2][0][0], data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
            row_data_column = ['s'+str(int(s))+'-pre', data[s][2][0][0], data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        for s in evaluation_station_id:
            row_data_column = ['s'+str(int(s))+'-ori', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
            row_data_column = ['s'+str(int(s))+'-pre', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
            for i in range((end_date - start_date).days + 1):
                row_data_column.append(0.0)
            pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    for p in tqdm(range(epochs), desc='Training'):
        net_dict = dict()
        gradient_dict = dict()
        for name, param in net.named_parameters():
            if param.requires_grad:
                gradient_dict[name] = torch.zeros(param.shape).to(device)
        for station in station_id:
            charging_sample = data[station][0]
            charging_lable = data[station][1]
            feature = data[station][2]
            date = data[station][3]
            weekday = data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday,divide_mode, divide_rate=0.8)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            net_dict[station] = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            temp_optimizer = torch.optim.Adam(net_dict[station].parameters(), lr=lr, weight_decay=1e-5)

            # plt.ion()
            for e in range(support_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday),dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = net_dict[station](input_data)
                    # output = net_dict[f](in_feature, nearby_Gaussian_covid)  # without sample
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()

            metrics_matrix=[0,0,0,0,0,0]
            num=0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num+=label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday),dim=1).to(device)
                temp_optimizer.zero_grad()
                output = net_dict[station](input_data)
                loss = temp_loss_function(output, label)
                loss.backward()
                temp_optimizer.step()
                RMSE, MAE, MAPE, MedAE, R2,  EVS = calculate_metrics(output.cpu().detach().numpy()*1, label.cpu().detach().numpy()*1)
                metrics_matrix[0] += RMSE * label.shape[0]
                metrics_matrix[1] += MAE * label.shape[0]
                metrics_matrix[2] += MAPE*label.shape[0]
                metrics_matrix[3] += MedAE * label.shape[0]
                metrics_matrix[4] += R2 * label.shape[0]
                metrics_matrix[5] += EVS * label.shape[0]
            for i in range(6):
                metrics_matrix[i]=metrics_matrix[i]/num
            if print_details:
                print('TRAIN on ID '+str(station)+': RMSE, MAE, MAPE, MedAE, R2,  EVS')
                print(metrics_matrix)

            for name, param in net_dict[station].named_parameters():
                if param.requires_grad:
                    gradient = copy.deepcopy(param.grad.data)
                    gradient_dict[name] += gradient

        for name, param in net.named_parameters():
            if param.requires_grad:
                param.grad = gradient_dict[name] / len(station_id)
        optimizer.step()
        optimizer.zero_grad()
        if print_details:
            print('--------------------------------------------------------------------------')
        total_matrix = [0, 0, 0, 0, 0,0]
        for station in evaluation_station_id:
            charging_sample = evaluation_data[station][0]
            charging_lable = evaluation_data[station][1]
            feature = evaluation_data[station][2]
            date = evaluation_data[station][3]
            weekday = evaluation_data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday, divide_mode,divide_rate=0.8,devide=False)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            temp_net = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            temp_optimizer = torch.optim.Adam(temp_net.parameters(), lr=lr, weight_decay=1e-5)

            for e in range(custom_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = temp_net(input_data)
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()
            metrics_matrix = [0, 0, 0, 0,0, 0]
            num = 0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num += label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                output = temp_net(input_data)
                RMSE, MAE, MAPE,MedAE, R2, EVS = calculate_metrics(output.cpu().detach().numpy()*1, label.cpu().detach().numpy()*1)
                metrics_matrix[0] += RMSE * label.shape[0]
                metrics_matrix[1] += MAE * label.shape[0]
                metrics_matrix[2] += MAPE * label.shape[0]
                metrics_matrix[3] += MedAE * label.shape[0]
                metrics_matrix[4] += R2 * label.shape[0]
                metrics_matrix[5] += EVS * label.shape[0]
                if p==epochs-1 and save_pre_result:
                    date_numpy=date.cpu().detach().numpy()
                    output_numpy=output.cpu().detach().numpy()
                    label_numpy=label.cpu().detach().numpy()
                    num_date=date_numpy.shape[0]
                    for l in range(num_date):
                        day=str(day_to_date(2023,date_numpy[l][0]))
                        pre_result_df.loc[pre_result_df['id'] == 's'+str(int(station))+'-ori', day] = label_numpy[l]
                        pre_result_df.loc[pre_result_df['id'] == 's'+str(int(station))+'-pre', day] = output_numpy[l]
            for i in range(6):
                metrics_matrix[i] = metrics_matrix[i] / num
                total_matrix[i]+=metrics_matrix[i]/len(evaluation_station_id)
            if print_details:
                print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
                print(metrics_matrix)
        if print_details:
            print('--------------------------------------------------------------------------')
        print('\n'+str(p)+'/'+str(epochs)+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS')
        print(total_matrix)
        log_file.writelines(
            str(p)+'/'+str(epochs)+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS'+ '\n'+ str(total_matrix)+'\n'
        )
        log_file.flush()
        if print_details:
            print('##########################################################################')
    if save_pre_result:
        for station in station_id:
            charging_sample = data[station][0]
            charging_lable = data[station][1]
            feature = data[station][2]
            date = data[station][3]
            weekday = data[station][4]
            train_sample, train_label, train_feature, train_date, train_weekday, test_sample, test_label, test_feature, test_date, test_weekday = divide_dataset(
                charging_sample, charging_lable, feature, date, weekday,divide_mode, divide_rate=0.8,devide=False)

            train_dataset = CreateDataset(train_sample, train_label, train_feature, train_date, train_weekday)
            test_dataset = CreateDataset(test_sample, test_label, test_feature, test_date, test_weekday)

            if batch_size is None:
                train_data_loader = DataLoader(train_dataset, batch_size=train_sample.shape[0], shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=test_sample.shape[0], shuffle=False)
            else:
                train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            temp_net = copy.deepcopy(net).to(device)
            temp_loss_function = torch.nn.MSELoss().to(device)
            temp_optimizer = torch.optim.Adam(temp_net.parameters(), lr=lr, weight_decay=1e-5)

            for e in range(custom_epochs):
                for i, ds in enumerate(train_data_loader):
                    sample, label, feature, date, weekday = ds
                    sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                    label = torch.reshape(label, [label.shape[0], 1]).to(device)
                    date = torch.reshape(date, [date.shape[0], 1])
                    weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                    input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                    temp_optimizer.zero_grad()
                    output = temp_net(input_data)
                    loss = temp_loss_function(output, label)
                    loss.backward()
                    temp_optimizer.step()
            num = 0
            for i, ds in enumerate(test_data_loader):
                sample, label, feature, date, weekday = ds
                num += label.shape[0]
                sample = torch.reshape(sample, [sample.shape[0], sample.shape[1]])
                label = torch.reshape(label, [label.shape[0], 1]).to(device)
                date = torch.reshape(date, [date.shape[0], 1])
                weekday = torch.reshape(weekday, [weekday.shape[0], 1])
                input_data = torch.cat((sample, feature, date, weekday), dim=1).to(device)
                output = temp_net(input_data)
                date_numpy=date.cpu().detach().numpy()
                output_numpy=output.cpu().detach().numpy()
                label_numpy=label.cpu().detach().numpy()
                num_date=date_numpy.shape[0]
                for l in range(num_date):
                    day=str(day_to_date(2023,date_numpy[l][0]))
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day] = label_numpy[l]
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day] = output_numpy[l]
        if not os.path.exists('/results/baselines/'):
            os.makedirs('/results/baselines/')
        if not os.path.exists('/results/baselines/'+baseline_name+ '/'):
            os.makedirs('/results/baselines/'+baseline_name+ '/')
        pre_result_df.to_csv('/results/baselines/'+baseline_name+ '/' + city_name+'_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + '.csv', index=False)

    if not os.path.exists('/results/model/baselines/' ):
        os.makedirs('/results/model/baselines/')
    if not os.path.exists('/results/model/baselines/' + baseline_name + '/'):
        os.makedirs('/results/model/baselines/' + baseline_name + '/')
    # torch.save(net.state_dict(),'model/baselines/' + baseline_name + '/Meta_Net_'+city_name+'_'+time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) +'.pt')



def baseline_arima_6days(data,evaluation_data, log_file,city_name='',baseline_name=''):

    station_id = list(data.keys())
    evaluation_station_id = list(evaluation_data.keys())
    all_id=station_id+evaluation_station_id
    start_date = datetime.date(2023,1,1)
    end_date = datetime.date(2023, 12, 31)
    temp_date = start_date
    data_column=['id','longitude','latitude']
    for i in range((end_date - start_date).days + 1):
        data_column.append(str(temp_date))
        temp_date += datetime.timedelta(days=1)
    pre_result_df=pd.DataFrame(columns=data_column)
    for s in station_id:
        row_data_column = ['s'+str(int(s))+'-ori', data[s][2][0][0], data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        row_data_column = ['s'+str(int(s))+'-pre', data[s][2][0][0], data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    for s in evaluation_station_id:
        row_data_column = ['s'+str(int(s))+'-ori', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        row_data_column = ['s'+str(int(s))+'-pre', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    total_matrix = [0, 0, 0, 0, 0,0]
    for station in station_id:
        charging_sample = data[station][0]
        charging_lable = data[station][1]
        temp_date = datetime.date(2023, 1, 1)
        predict_values_list=[]
        for d_charging in range(365):
            d_sample=charging_sample[d_charging]
            d_lable=charging_lable[d_charging]
            time_series = d_sample.flatten()
            model = pm.auto_arima(time_series, seasonal=False)
            predict_values = model.predict(n_periods=1)
            day_str = str(temp_date)
            pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] = d_lable
            pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day_str] = predict_values[0]
            temp_date += datetime.timedelta(days=1)
            predict_values_list.append(predict_values[0])
        RMSE, MAE, MAPE, MedAE, R2, EVS = calculate_metrics(np.array(predict_values_list),charging_lable)
        print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
        print([RMSE, MAE, MAPE, MedAE, R2, EVS])
        R2 = R2 if not np.isnan(R2) else 0
        EVS = EVS if not np.isnan(EVS) else 0
        total_matrix[0] += RMSE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[1] += MAE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[2] += MAPE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[3] += MedAE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[4] += R2/ (len(station_id)+len(evaluation_station_id))
        total_matrix[5] += EVS/ (len(station_id)+len(evaluation_station_id))
    for station in evaluation_station_id:
        metrics_matrix = [0, 0, 0, 0,0, 0]
        charging_sample = evaluation_data[station][0]
        charging_lable = evaluation_data[station][1]
        temp_date = datetime.date(2023, 1, 1)
        predict_values_list=[]
        for d_charging in range(365):
            d_sample=charging_sample[d_charging]
            d_lable=charging_lable[d_charging]
            time_series = d_sample.flatten()
            model = pm.auto_arima(time_series, seasonal=False)
            predict_values = model.predict(n_periods=1)
            day_str = str(temp_date)
            pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] = d_lable
            pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day_str] = predict_values
            temp_date += datetime.timedelta(days=1)
            predict_values_list.append(predict_values[0])
        RMSE, MAE, MAPE, MedAE, R2, EVS = calculate_metrics(np.array(predict_values_list),charging_lable)
        print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
        print([RMSE, MAE, MAPE, MedAE, R2, EVS])
        total_matrix[0] += RMSE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[1] += MAE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[2] += MAPE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[3] += MedAE/ (len(station_id)+len(evaluation_station_id))
        total_matrix[4] += R2/ (len(station_id)+len(evaluation_station_id))
        total_matrix[5] += EVS/ (len(station_id)+len(evaluation_station_id))

    print('\n'+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS')
    print(total_matrix)
    log_file.writelines(
        'AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS' + '\n' + str(total_matrix) + '\n'
    )
    log_file.flush()
    if not os.path.exists('/results/baselines/'):
        os.makedirs('/results/baselines/')
    if not os.path.exists('/results/baselines/' + baseline_name + '/'):
        os.makedirs('/results/baselines/' + baseline_name + '/')
    pre_result_df.to_csv('/results/baselines/' + baseline_name + '/' + city_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()) + '.csv',index=False)



def baseline_arima_monthly(data,evaluation_data, log_file, city_name='',baseline_name=''):

    station_id = list(data.keys())
    evaluation_station_id = list(evaluation_data.keys())
    all_id=station_id+evaluation_station_id
    start_date = datetime.date(2023,1,1)
    end_date = datetime.date(2023, 12, 31)
    temp_date = start_date
    data_column=['id','longitude','latitude']
    for i in range((end_date - start_date).days + 1):
        data_column.append(str(temp_date))
        temp_date += datetime.timedelta(days=1)
    pre_result_df=pd.DataFrame(columns=data_column)
    for s in station_id:
        row_data_column = ['s'+str(int(s))+'-ori', data[s][2][0][0], data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        row_data_column = ['s'+str(int(s))+'-pre', data[s][2][0][0], data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    for s in evaluation_station_id:
        row_data_column = ['s'+str(int(s))+'-ori', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
        row_data_column = ['s'+str(int(s))+'-pre', evaluation_data[s][2][0][0], evaluation_data[s][2][0][1]]
        for i in range((end_date - start_date).days + 1):
            row_data_column.append(0.0)
        pre_result_df.loc[len(pre_result_df.index)] = row_data_column
    total_matrix = [0, 0, 0, 0, 0,0]
    for station in station_id:
        charging_lable = data[station][1]
        time_series = charging_lable.flatten()
        for month in range(1, 13, 1):
            input_data = []
            predict_index=0
            for input_month_id in range(2):
                input_month=month+input_month_id
                if input_month>12:
                    input_month=input_month-12
                    days_in_month = (datetime.date(2023, input_month + 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                elif input_month==12:
                    days_in_month = (datetime.date(2024, 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                else:
                    days_in_month = (datetime.date(2023, input_month + 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                predict_index=start_index + days_in_month
            try:
                model = pm.auto_arima(input_data, seasonal=False)
                days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                predict_month=month+2
                if predict_month>12:
                    predict_month=predict_month-12
                predict_values = model.predict(n_periods=days_in_month[predict_month-1])
            except:
                predict_values = [sum(input_data)/len(input_data) for _ in range(days_in_month[predict_month-1])]
            temp_date = datetime.date(2023, predict_month, 1)
            label_list=[]
            for i in range(days_in_month[predict_month-1]):
                day_str = str(temp_date)
                if predict_index+i<365:
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] = charging_lable[predict_index+i]
                    label_list.append(charging_lable[predict_index+i])
                else:
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] =charging_lable[predict_index + i-365]
                    label_list.append(charging_lable[predict_index + i-365])
                pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day_str] = predict_values[i]
                temp_date += datetime.timedelta(days=1)
            RMSE, MAE, MAPE, MedAE, R2, EVS = calculate_metrics(np.array(predict_values), np.array(label_list))
            print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
            print([RMSE, MAE, MAPE, MedAE, R2, EVS])
            R2 = R2 if not np.isnan(R2) else 0
            EVS = EVS if not np.isnan(EVS) else 0
            total_matrix[0] += RMSE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[1] += MAE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[2] += MAPE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[3] += MedAE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[4] += R2 * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[5] += EVS * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
    for station in evaluation_station_id:
        charging_lable = evaluation_data[station][1]
        time_series = charging_lable.flatten()
        for month in range(1, 13, 1):
            input_data = []
            predict_index=0
            for input_month_id in range(2):
                input_month=month+input_month_id
                if input_month>12:
                    input_month=input_month-12
                    days_in_month = (datetime.date(2023, input_month + 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                elif input_month==12:
                    days_in_month = (datetime.date(2024, 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                else:
                    days_in_month = (datetime.date(2023, input_month + 1, 1) - datetime.date(2023, input_month,1)).days
                    start_index = (datetime.date(2023, input_month, 1) - datetime.date(2023, 1, 1)).days
                    input_data.extend(time_series[start_index:start_index + days_in_month])
                predict_index=start_index + days_in_month
            try:
                model = pm.auto_arima(input_data, seasonal=False)
                days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
                predict_month = month + 2
                if predict_month > 12:
                    predict_month = predict_month - 12
                predict_values = model.predict(n_periods=days_in_month[predict_month - 1])
            except:
                predict_values = [sum(input_data) / len(input_data) for _ in range(days_in_month[predict_month - 1])]
            temp_date = datetime.date(2023, predict_month, 1)
            label_list=[]
            for i in range(days_in_month[predict_month-1]):
                day_str = str(temp_date)
                if predict_index+i<365:
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] = charging_lable[predict_index+i]
                    label_list.append(charging_lable[predict_index+i])
                else:
                    pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-ori', day_str] =charging_lable[predict_index + i-365]
                    label_list.append(charging_lable[predict_index + i-365])
                pre_result_df.loc[pre_result_df['id'] == 's' + str(int(station)) + '-pre', day_str] = predict_values[i]
                temp_date += datetime.timedelta(days=1)
            RMSE, MAE, MAPE, MedAE, R2, EVS = calculate_metrics(np.array(predict_values), np.array(label_list))
            print('TEST on ID ' + str(station) + ': RMSE, MAE, MAPE, MedAE, R2,  EVS')
            print([RMSE, MAE, MAPE, MedAE, R2, EVS])
            R2 = R2 if not np.isnan(R2) else 0
            EVS = EVS if not np.isnan(EVS) else 0
            total_matrix[0] += RMSE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[1] += MAE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[2] += MAPE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[3] += MedAE * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[4] += R2 * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))
            total_matrix[5] += EVS * days_in_month[predict_month-1] /365 / (len(station_id) + len(evaluation_station_id))



    print('\n'+'-AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS')
    print(total_matrix)
    log_file.writelines(
        'AVG TEST: RMSE, MAE, MAPE, MedAE, R2,  EVS' + '\n' + str(total_matrix) + '\n'
    )
    log_file.flush()
    if not os.path.exists('/results/baselines/'):
        os.makedirs('/results/baselines/')
    if not os.path.exists('/results/baselines/' + baseline_name + '/'):
        os.makedirs('/results/baselines/' + baseline_name + '/')
    pre_result_df.to_csv('/results/baselines/' + baseline_name + '/' + city_name + '_' + time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime()) + '.csv',index=False)