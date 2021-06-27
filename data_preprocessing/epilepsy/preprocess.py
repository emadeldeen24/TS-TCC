import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data.csv")
y = data.iloc[:,-1]
x = data.iloc[:,1:-1]

x = x.to_numpy()
y = y.to_numpy()
y = y -1
scaler = MinMaxScaler()
x = scaler.fit_transform(x)

for i,j in enumerate(y):
    if j != 0:
        y[i] = 1



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, "train.pt")

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test).unsqueeze(1)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, "test.pt")
