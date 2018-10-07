import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import accuracy_score,mean_squared_error
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder   #encoding the gender data
lb1 = LabelEncoder()
data = pd.read_csv(r"C:\Users\Madhura\Desktop\MyProject\IBM Hack Challenge\CompleteData.csv")
data.emotion = lb1.fit_transform(data.emotion)
data.gender = lb1.fit_transform(data.gender)
x = data.iloc[:,0:4].values
y = data.iloc[:,-1].values


x_train = x[:30000]
x_test = x[30000:]
y_train = y[:30000]
y_test = y[30000:]

teacher = KNeighborsClassifier(n_neighbors=3)
learner = teacher.fit(x_train,y_train)

Yp = learner.predict(x_test)
compare = pd.DataFrame({"Actual":y_test, "Predicted":Yp})
compare

acc = (accuracy_score(y_test,Yp)*100)
print(acc)
