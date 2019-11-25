import pandas as pd
msg=pd.read_csv('p6.csv',names=['message','label'])
msg=pd.get_dummies(data=msg,columns=['label'])

X=msg.message
y=msg.iloc[:,-1]
#splitting the dataset into train and test data
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,y)

#output of count vectoriser is a sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
xtrain_dtm = count_vect.fit_transform(xtrain)
xtest_dtm=count_vect.transform(xtest)
print(count_vect.get_feature_names())
df=pd.DataFrame(xtrain_dtm.toarray(),columns=count_vect.get_feature_names())
print(df)#tabular representation
print(xtrain_dtm) #sparse matrix representation
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(xtrain_dtm,ytrain)
predicted = clf.predict(xtest_dtm)
#printing accuracy metrics
from sklearn import metrics
print('Accuracy \n',metrics.accuracy_score(ytest,predicted))
print('Confusion matrix \n',metrics.confusion_matrix(ytest,predicted))
print('Recall \n ',metrics.recall_score(ytest,predicted))
print('Precision \n ',metrics.precision_score(ytest,predicted))