import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('heart1.csv')
df = df.drop_duplicates()
#print(df.shape)
X = np.array(df.iloc[:, 0:11])
y = np.array(df.iloc[:, 11:])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y.reshape(-1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(X_train,y_train)
# y_preden=sv.predict(X_test)
# accuracy_4m=accuracy_score(y_test,y_preden)*100
# print('svm accuracy : ',accuracy_4m)
from sklearn.ensemble import VotingClassifier
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(random_state=42)
clf3 = GaussianNB()
clf4 = SVC(kernel='linear',probability=True)
clf5 = DecisionTreeClassifier()

eclf = VotingClassifier(estimators=[('LR', clf1), ('RF', clf2), ('GNB', clf3), ('SVC', clf4),('DT',clf5)],
                        voting='soft') 


eclf.fit(X_train, y_train)
y_predens=eclf.predict(X_test)
accuracy_4m=accuracy_score(y_test,y_predens)*100
print('ensemble : ',accuracy_4m)

pickle.dump(eclf, open('mypk.pkl', 'wb'))