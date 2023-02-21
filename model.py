import pandas
import pandas as pd
import numpy as np
data = pd.read_csv('/Users/qiuhaoxuan/PycharmProjects/数据分析/titanic/train.csv')
test_data = pd.read_csv('/Users/qiuhaoxuan/PycharmProjects/数据分析/titanic/test.csv')
#%%
data:pandas.DataFrame
data=data.drop(labels='Name',axis=1)
data=data.drop(labels='Ticket',axis=1)
data=data.drop(labels='Cabin',axis=1)
print(np.sum(data.isnull()))
dit_Embarked = {'C':1,'S':2,'Q':3}
dit_sex = {'female':1,'male':0}
data['Age'].fillna(data['Age'].mean(),inplace=True)
data=data.dropna()
data.index=np.arange(len(data))
for i in range(len(data)):
    if data['Sex'][i] in dit_sex.keys() and data['Embarked'][i] in dit_Embarked.keys():
        data['Sex'][i] = dit_sex[data['Sex'][i]]
        data['Embarked'][i] = dit_Embarked[data['Embarked'][i]]
    else:
        continue
#%%
test_data:pandas.DataFrame
test_data=test_data.drop(labels='Name',axis=1)
test_data=test_data.drop(labels='Ticket',axis=1)
test_data=test_data.drop(labels='Cabin',axis=1)
print(np.sum(test_data.isnull()))
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)
dit_Embarked = {'C':1,'S':2,'Q':3}
dit_sex = {'female':1,'male':0}
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
test_data.index=np.arange(len(test_data))
for i in range(len(test_data)):
    if test_data['Sex'][i] in dit_sex.keys() and test_data['Embarked'][i] in dit_Embarked.keys():
        test_data['Sex'][i] = dit_sex[test_data['Sex'][i]]
        test_data['Embarked'][i] = dit_Embarked[test_data['Embarked'][i]]
    else:
        continue
#%%
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# x = data.iloc[:,2:]
# x = x.values
# y = data["Survived"]
# y = y.values
# x_test = test_data.iloc[:,1:]
# x_test = x_test.values
# adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),learning_rate=0.8,random_state=5)
# adaboost.fit(x,y)
# y_p = adaboost.predict(x_test)
# result = pd.DataFrame()
# result['PassengerId'] = test_data['PassengerId']
# result['Survived'] = y_p
# result:pd.DataFrame
# result.to_csv('result.csv')

#%%
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.preprocessing import StandardScaler
standard = StandardScaler()
x = data.iloc[:,2:]
x = x.values
y = data["Survived"]
y = y.values
x_test = test_data.iloc[:,1:]
x_test = x_test.values
x_test = standard.fit_transform(x_test)
x = standard.fit_transform(x)
x_test = standard.fit_transform(x_test)
x_test = np.array(x_test,dtype=np.float32)
x = np.array(x,dtype=np.float32)
x = torch.from_numpy(x).type(torch.Tensor)
y = np.array(y,dtype=np.float32)
y = torch.from_numpy(y).type(torch.Tensor)
y=torch.tensor(y,dtype=torch.long)
train_data = Data.TensorDataset(x,y)
data_loader = Data.DataLoader(dataset=train_data,batch_size=400)

class logisticmodel(nn.Module):
    def __init__(self):
        super(logisticmodel, self).__init__()
        self.model = nn.Sequential(nn.Linear(7,150),
                                   nn.Sigmoid(),
                                   nn.Linear(150,7))
    def forward(self,x):
        return self.model(x)
logistic = logisticmodel()
opt = torch.optim.Adam(logistic.parameters(),lr=1)
loss = nn.CrossEntropyLoss()
l_before=0
for i in range(21400):
    if i % 200==0 and i>0:
        print(l.item(),i)
    for j,(x,y) in enumerate(data_loader):
        y_p = logistic(x)
        l = loss(y_p,y)
        if i % 400 == 0 and i > 0 and l.item()<l_before:
            print("save model")
            torch.save(logistic.state_dict(),"model.pth")
        opt.zero_grad()
        l.backward()
        opt.step()
    l_before = l.item()
    if 0.3 > l.item() > 0.2:
        for param in opt.param_groups:
            param['lr'] = 0.8
    elif 0.2 > l.item() > 0.11:
        for param in opt.param_groups:
            param['lr'] = 0.1
    elif 0.11 > l.item() > 0.08:
        for param in opt.param_groups:
            param['lr'] = 0.05
    elif 0.08 > l.item() > 0.06:
        for param in opt.param_groups:
            param['lr'] = 0.035
    elif 0.06 > l.item():
        for param in opt.param_groups:
            param['lr'] = 0.015
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_p = logistic(x_test)
result = torch.argmax(y_p,axis=1)
result = pd.DataFrame(result,columns=['Survived'])
result['PassengerId'] = test_data['PassengerId']
result.to_csv('result.csv')