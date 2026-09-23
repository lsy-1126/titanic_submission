import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(0)

train_df = pd.read_csv('train.csv')

train_df['Age']=train_df['Age'].fillna(train_df['Age'].median())
train_df['Fare']=train_df['Fare'].fillna(train_df['Fare'].median())
train_df['Embarked']=train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])
train_df['Sex']=train_df['Sex'].map({'male': 0, 'female': 1})
train_df=pd.get_dummies(train_df, columns=['Embarked'], prefix='Embarked')
train_df.drop(['Name','Ticket','Cabin','PassengerId','SibSp','Parch'],axis=1, inplace=True)

X = train_df.drop('Survived', axis=1)
y = train_df['Survived']

def split(X,y,test_size=0.2,random_state=0):
    np.random.seed(random_state)
    indices=np.arange(len(X))
    np.random.shuffle(indices)
    
    split_idx=int(len(X)*(1-test_size))
    train_indices=indices[:split_idx]
    test_indices=indices[split_idx:]
    
    return (X.iloc[train_indices],X.iloc[test_indices], 
            y.iloc[train_indices],y.iloc[test_indices])

X_train,X_test,y_train,y_test=split(X,y,test_size=0.2,random_state=0)

train_mean=X_train.mean()
train_std=X_train.std()
X_train_scaled=(X_train-train_mean)/train_std
X_test_scaled=(X_test-train_mean)/train_std 

with open('preprocess_params.pkl','wb') as f:
    pickle.dump({'mean':train_mean,'std':train_std,'columns':X.columns.tolist()}, f)


class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values,dtype=torch.float32)
        self.labels = torch.tensor(labels.values,dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        return self.features[idx], self.labels[idx]
batch_size=32
train_dataset=TitanicDataset(X_train_scaled,y_train)
test_dataset=TitanicDataset(X_test_scaled,y_test)
train_loader=DataLoader(train_dataset,batch_size=batch_size, shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

class SimplifiedTitanicModel(nn.Module):
    def __init__(self,input_dim):
        super(SimplifiedTitanicModel,self).__init__()              
        self.linear = nn.Linear(input_dim,1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        x=self.linear(x)
        x=self.sigmoid(x)
        return x

input_dim=X_train_scaled.shape[1]
model=SimplifiedTitanicModel(input_dim)

criterion=nn.BCELoss() 
optimizer=optim.Adam(model.parameters(),lr=0.01)

epochs=100
train_losses=[]
train_accuracies=[]
test_accuracies=[]

for epoch in range(epochs):
    model.train()
    running_loss=0.0
    correct_train=0
    total_train=0
    
    for features, labels in train_loader:
        optimizer.zero_grad()
        outputs=model(features)
        loss=criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss+=loss.item()
        predicted=(outputs >=0.5).float()
        correct_train+=(predicted==labels).sum().item()
        total_train+=labels.size(0)
        
    epoch_loss=running_loss/len(train_loader)
    train_acc=correct_train/total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(train_acc)
    
    model.eval()
    correct_test=0
    total_test=0
    with torch.no_grad():
        for features,labels in test_loader:
            outputs=model(features)
            predicted=(outputs >= 0.5).float()
            correct_test+=(predicted==labels).sum().item()
            total_test+=labels.size(0)
    
    test_acc=correct_test/total_test
    test_accuracies.append(test_acc)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_losses, label='Training Loss', color='blue')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(False)
plt.savefig('loss_curve.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy', color='green')
plt.plot(range(1, epochs+1), test_accuracies, label='Test Accuracy', color='red')
plt.title('Train and Test Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(False)
plt.savefig('accuracy_curve.png')
plt.show()

torch.save(model.state_dict(), 'titanic_model.pth')





def predict_new_passenger(passenger_info):
    with open('preprocess_params.pkl', 'rb') as f:
        params = pickle.load(f)

    new_df = pd.DataFrame([passenger_info])
    new_df['Age'] = new_df['Age'].fillna(params['mean']['Age'])
    new_df['Fare'] = new_df['Fare'].fillna(params['mean']['Fare'])
    new_df['Embarked'] = new_df['Embarked'].fillna('S')
    new_df['Sex'] = new_df['Sex'].map({'male': 0, 'female': 1})
    new_df = pd.get_dummies(new_df, columns=['Embarked'], prefix='Embarked')
    new_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'], 
                axis=1, inplace=True, errors='ignore')
    new_df = new_df.reindex(columns=params['columns'], fill_value=0)
    new_scaled = (new_df - params['mean']) / params['std']
    model.eval()
    with torch.no_grad():
        tensor_input = torch.tensor(new_scaled.values, dtype=torch.float32)
        output = model(tensor_input)
        prediction = (output >= 0.5).int().item()
    
    return prediction

new_passenger = {      #假信息
    'Pclass': 3,
    'Name': 'L,jb',
    'Sex': 'male',
    'Age': 18.0,
    'SibSp': 1,
    'Parch': 0,
    'Ticket': 'A/1 00001',
    'Fare': 10.0,
    'Cabin': np.nan,
    'Embarked': 'S'
}
result = predict_new_passenger(new_passenger)
print(f"ljb预测结果: {'幸存' if result == 1 else '遇难'} ({result})")