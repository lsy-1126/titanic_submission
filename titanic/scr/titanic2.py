import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn 
import torch.optim as optim
import pickle
import random
from torch.utils.data import Dataset, DataLoader

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

train_df = pd.read_csv('train.csv')

train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare']=train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['family']=train_df['SibSp']+train_df['Parch']
train_df['cabin']=train_df['Cabin'].notnull().astype(int)
train_df=pd.get_dummies(train_df,columns=['Embarked'],prefix=['Embarked'])
train_df['Sex']=train_df['Sex'].map({'male': 1, 'female': 0})
train_df['Title'] = train_df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
train_df['Title'] = train_df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'], 'Rare')
train_df['Title'] = train_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_df['Title'] = train_df['Title'].replace('Mme', 'Mrs')
train_df=pd.get_dummies(train_df,columns=['Title'],prefix=['Title'])
train_df.drop(['Name', 'Ticket' , 'PassengerId', 'SibSp', 'Parch', 'Cabin'],axis=1,inplace=True)


X=train_df.drop(['Survived'],axis=1)
Y=train_df['Survived']

set_seed(0)
def train_test_split(X,Y,train_size=0.8):
        induces=np.random.permutation(len(X))
        split_idx=int(len(X)*0.8)
        train_idx=induces[:split_idx]
        test_idx=induces[split_idx:]
        return X.iloc[train_idx],X.iloc[test_idx],Y.iloc[train_idx],Y.iloc[test_idx]
train_X,test_X,train_Y,test_Y=train_test_split(X,Y,train_size=0.8)

train_mean=train_X.mean()
train_std=train_X.std()
train_std[train_std==0]=1
train_X_scaled=(train_X-train_mean)/train_std
test_X_scaled=(test_X-train_mean)/train_std

with open('data.pkl','wb') as f:
    pickle.dump({'mean': train_mean,'std': train_std,'columns': X.columns.tolist()}, f)
    
def run_experiment(seed,train_X_scaled,train_Y,test_X_scaled,test_Y):
    set_seed(seed)

    class Titanic(Dataset):
        def __init__(self,X,Y):
            self.X=torch.tensor(X.values,dtype=torch.float32)
            self.Y=torch.tensor(Y.values,dtype=torch.float32).unsqueeze(1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self,idx):
            return self.X[idx],self.Y[idx]

    train_dataset=Titanic(train_X_scaled,train_Y)
    test_dataset=Titanic(test_X_scaled,test_Y)
    train_loader=DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_loader=DataLoader(test_dataset,batch_size=32,shuffle=False)


    class Titanic_model(nn.Module):
        def __init__(self,input_dim):
            super(Titanic_model,self).__init__()
            self.fc1=nn.Linear(input_dim,64)
            self.fc2=nn.Linear(64,32)
            self.fc3=nn.Linear(32,1)
            self.relu=nn.ReLU()
            self.dropout=nn.Dropout(0.2)
            self.sigmoid=nn.Sigmoid()

        def forward(self,X):
            X=self.fc1(X)
            X=self.relu(X)
            X=self.dropout(X)
            X=self.fc2(X)
            X=self.fc3(X)
            X=self.sigmoid(X)
            return X

    input_dim=train_X_scaled.shape[1]
    model=Titanic_model(input_dim)

    criterion=nn.BCELoss()
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.75,patience=5,mode='min')

    epochs=100
    train_losses=[]
    train_accuracies=[]
    test_losses=[]
    test_accuracies=[]
    best_accuracy = 0  
    best_epoch = 0
    for epoch in range(epochs):
        model.train()
        running_loss=0
        correct_train=0
        total_train=0
        for features,labels in train_loader:
            optimizer.zero_grad()
            output=model(features)
            loss=criterion(output,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            prediction=(output>0.5).float()
            correct_train+=(prediction==labels).sum().item()
            total_train+=labels.shape[0]
        train_accuracy=correct_train/total_train
        train_loss=running_loss/len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        model.eval()
        correct_test = 0
        total_test = 0
        test_running_loss = 0   
        with torch.no_grad():
            for features,labels in test_loader:
                    output=model(features)
                    loss=criterion(output,labels)
                    test_running_loss+=loss.item()
                    prediction=(output>0.5).float()
                    correct_test+=(prediction==labels).sum().item()
                    total_test+=labels.shape[0]
        test_accuracy=correct_test/total_test
        test_loss=test_running_loss/len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        if  test_accuracy>best_accuracy:
            best_accuracy=test_accuracy
            best_epoch=epoch+1
            torch.save(model.state_dict(),'best_model.pth')
            print(f"epoch:{epoch+1},better model,accuracy={best_accuracy:.4f}")

    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    print(f"种子{seed}的最佳验证集准确率:{best_accuracy:.4f}")
    test_X_tensor=torch.tensor(test_X_scaled.values, dtype=torch.float32)
    with torch.no_grad():
        predictions=model(test_X_tensor).numpy().flatten()
    return predictions,best_accuracy,train_losses,test_losses,train_accuracies,test_accuracies

seeds=[0,1,2,3,4]

predictions=[]
accuracies=[]
all_train_losses=[]
all_test_losses=[]
all_train_accuracies=[]
all_test_accuracies=[]
for seed in seeds:
    preds,acc,trainloss,testloss,trainacc,testacc=run_experiment(seed,train_X_scaled,train_Y,test_X_scaled,test_Y)
    predictions.append(preds)
    accuracies.append(acc)
    all_train_losses.append(trainloss)
    all_test_losses.append(testloss)
    all_train_accuracies.append(trainacc)
    all_test_accuracies.append(testacc)
predictions=np.mean(predictions,axis=0)
final_predictions=(predictions>0.5).astype(int)
final_accuracy=(final_predictions==test_Y.values).sum()/len(test_Y)
atrain_loss = np.mean(all_train_losses, axis=0)
atrain_accuracy = np.mean(all_train_accuracies, axis=0)
atest_accuracy = np.mean(all_test_accuracies, axis=0)
print(f"\n*** 最终结果 ***")
print(f"集成模型验证集准确率: {final_accuracy:.4f}")

plt.figure(figsize=(10,5))
plt.plot(range(1,101),atrain_loss,label='Average Training Loss',color='black')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_curve.png')
plt.show()

plt.figure(figsize=(10,5))
plt.plot(range(1,101),atrain_accuracy,label='Average Training Accuracy',color='black')
plt.plot(range(1,101),atest_accuracy,label='Average Test Accuracy',color='red')
plt.axhline(y=final_accuracy,color='blue',linestyle='--',label=f'Ensemble Accuracy({final_accuracy:.4f})')
plt.title('Train and Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png')
plt.show()



    









