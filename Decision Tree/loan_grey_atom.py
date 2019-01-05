

import numpy as np
import pandas as pd

# Importing the dataset
loan = pd.read_csv('C:/Excellence/Grey_Atom/Decision_Tree/data/loan_prediction.csv')
###  Viewing the Data  ###############################

loan.head()
loan.groupby('Loan_Amount_Term').count()
loan.groupby('Loan_Status').count()
loan.groupby('Credit_History').count()
loan.isnull().sum(axis=0)


X = loan.iloc[:, :5].values
y = loan.iloc[:, 5:].values


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier = DecisionTreeClassifier(criterion = 'entropy', 
                                    max_depth=5, 
                                    min_samples_split=107, 
                                    min_samples_leaf=10, 
                                    max_features=None,  
                                    max_leaf_nodes=4,
                                    random_state = 0)
 
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm



#######################################################################################
from sklearn.externals.six import StringIO  
from sklearn import tree
from sklearn.tree import export_graphviz
from IPython.display import Image  
import pydotplus
#######################################################################################
dot_data = StringIO()
export_graphviz(classifier, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_pdf('loan_tree4.pdf')
#!dot -Tpng tree.dot -o tree.png -Gdpi=600
#pwd
Image(graph.create_png())




















