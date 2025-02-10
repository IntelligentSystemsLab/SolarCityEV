from sklearn import svm
import copy

def oneclassSVM(data,average_index = 3):
    clf = svm.OneClassSVM(nu=0.05, kernel="rbf")
    clf.fit(data)
    y_pred = clf.predict(data)
    output_data = copy.deepcopy(data)

    for i in range(data.shape[0]):
        temp = 0
        num = 0
        if y_pred[i] == -1:
            for j in range(average_index):
                if i + j >= data.shape[0]:
                    temp += data[i-j]
                    num += 1
                elif i - j < 0:
                    temp += data[i+j]
                    num += 1
                else:
                    temp += data[i+j] + data[i-j]
                    num += 2

            output_data[i] = temp / num

    return output_data

def Moving_Average_Smooth(data,average_index = 3):
    output_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        temp = 0
        num = 0
        for j in range(average_index):
            if i + j >= data.shape[0]:
                temp += data[i-j]
                num += 1
            elif i - j < 0:
                temp += data[i+j]
                num += 1
            else:
                temp += data[i+j] + data[i-j]
                num += 2
        temp_num=  temp / num

        output_data[i] = temp_num

    return output_data

def Moving_Average_Smooth_range(data,rang,average_index = 3,rate=1):
    output_data = copy.deepcopy(data)
    for i in range(data.shape[0]):
        if i in rang:
            temp = 0
            num = 0
            for j in range(average_index):
                if i + j >= data.shape[0]:
                    temp += data[i-j]
                    num += 1
                elif i - j < 0:
                    temp += data[i+j]
                    num += 1
                else:
                    temp += data[i+j] + data[i-j]
                    num += 2
            temp_num=  temp / num

            output_data[i] = temp_num*rate

    return output_data
