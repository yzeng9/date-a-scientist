import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn import preprocessing
#Create your df here:
df=pd.read_csv("profiles.csv")
#print(df.essay6.head())

#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16, 80)
#plt.show()

#body_dict=df.body_type.value_counts()
#x=np.arange(len(body_dict.keys()))
#body_types=[]
#for k in body_dict.keys():
#    type=body_dict[k]
#    body_types.append(type)

#plt.barh(x, body_types)
#plt.yticks(x, body_dict.keys())
#plt.title("Body Types")
#plt.show()

#plt.hist(df['income'], bins=20, rwidth=0.75)
#plt.xlabel("income")
#plt.ylabel("frequency")
#plt.xlim([20000, 1000000])
#plt.show()


def calculate_ave(x):
    word_list=x.split()
    total_length=0
    for word in word_list:
        word_len=len(word)
        total_length+=word_len
    if len(word_list)!=0:
        ave_len=total_length/len(word_list)
        return ave_len
    else:
        return 0

def convert_to_num(type_list):
    num_list=[]
    t_dict={"rather not say": 1, "used up": 2, "jacked": 3, "overweight": 4, "full figured": 5, "skinny": 6, "a little extra": 7, "curvy": 8, "thin": 9, "athletic": 10, "fit": 11, "average":12}
    for type in type_list:
        num=t_dict[type]
        num_list.append(num)
    return num_list
### data augmentation
drink_mapping= {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinking_code"]=df.drinks.map(drink_mapping)

#print(df.drugs.value_counts())
drug_mapping={"never": 0, "sometimes": 1, "often": 3}
df["drug_code"]=df.drugs.map(drug_mapping)

test_cols=['body_type','drinking_code','drug_code']
all_cols=df.dropna(subset=test_cols)
#print(all_cols.drugs.head())

###normalization
normalized_cols=all_cols[['drinking_code', 'drug_code']]
x=normalized_cols.values
min_max_scaler=preprocessing.MinMaxScaler()
x_scaled=min_max_scaler.fit_transform(x)

normalized_cols=pd.DataFrame(x_scaled, columns=normalized_cols.columns)

drink_drug_data=normalized_cols[['drinking_code','drug_code']]
body_target=all_cols['body_type']


essay_cols=["essay2", "essay6"]
both_essays=df[essay_cols].replace(np.nan, '', regex=True)
both_essays=both_essays[essay_cols].apply(lambda x: ' '.join(x), axis=1)
#print(both_essays.head())
df["average_word_length"]=both_essays.apply(lambda x: calculate_ave(x))
#print(df["average_word_length"].head())
df["income_adjusted"]=df['income'].replace(-1, np.nan)
income_cols=df[['income_adjusted','average_word_length']]
income_cols=income_cols.dropna(subset=['income_adjusted'])
#print(income_cols['income_adjusted'].head())

normalized_income_cols=income_cols
m=normalized_income_cols.values
min_max_scaler_2=preprocessing.MinMaxScaler()
m_scaled=min_max_scaler_2.fit_transform(m)

normalized_income_cols=pd.DataFrame(m_scaled, columns=normalized_income_cols.columns)
income_data=normalized_income_cols['income_adjusted']
word_len_data=normalized_income_cols['average_word_length']




### classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC

[training_data, validation_data, training_labels, validation_labels]=train_test_split(drink_drug_data, body_target, test_size=0.2, random_state=50
)

#k-neaest neighbor classification
"""
k_list=[]
accuracies=[]
for k in range(1,51):
    k_list.append(k)
    classifier=KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)
    accuracies.append(classifier.score(validation_data, validation_labels))

plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("validation accuracy")
plt.title("body type claasifier accuracy")
plt.show()
"""

"""
classifier=KNeighborsClassifier(n_neighbors=10)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))
label_list=convert_to_num(validation_labels)
result_list=convert_to_num(classifier.predict(validation_data))
print(precision_score(label_list, result_list, average='macro'))
print(recall_score(label_list, result_list, average='micro'))
"""
"""
#support vector machines classifier
classifier_2=SVC(kernel='rbf', gamma=0.5, C=2)
classifier_2.fit(training_data, training_labels)
print(classifier_2.score(validation_data, validation_labels))
label_list=convert_to_num(validation_labels)
result_list=convert_to_num(classifier_2.predict(validation_data))
print(precision_score(label_list, result_list, average='micro'))
print(recall_score(label_list, result_list, average='micro'))
"""

### regression

from sklearn.linear_model import LinearRegression
from sklearn import svm

word_len_data=word_len_data.values.reshape(-1,1)
[training_data, validation_data, training_targets, validation_targets]=train_test_split(word_len_data, income_data, test_size=0.2, random_state=50
)

regr1=LinearRegression()
regr1.fit(training_data, training_targets)
income_predicted=regr1.predict(validation_data)
print(regr1.coef_)

"""
plt.plot(validation_data, income_predicted)
plt.scatter(validation_data, validation_targets)
plt.xlabel("scaled_average_word_length")
plt.ylabel("scaled_income")
plt.show()
"""

"""
regr2=svm.SVR()
regr2.fit(training_data, training_targets)
income_predicted=regr2.predict(validation_data)
plt.plot(validation_data, income_predicted)
plt.scatter(validation_data, validation_targets)
plt.xlabel("scaled_average_word_length")
plt.ylabel("scaled_income")
plt.show()
"""
