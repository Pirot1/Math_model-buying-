#гипотеза влияет ли уровень знаний на покупку?
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')

#гипотеза

temp = df[df['   result']==1]['education_status'].value_counts()
dpi = 100
fig = plt.figure(dpi = dpi, figsize = (1000 / dpi, 600 / dpi) )#Размер + размер текста

temp.plot(kind = 'pie', label = '', title = 'Как вы видете Alumnus (Specialist) больше всех покупают курсы',autopct='%.1f',radius = 1.5)#autopct='%.1f' = задаёт проценты
plt.show()

#clear

df.drop(['bdate','id','has_photo','city','followers_count','occupation_name','last_seen','relation','people_main','life_main','graduation','career_end','career_start','has_mobile'],axis = 1, inplace = True)

#sex

def sex_apply(sex):
    if sex ==1:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)

#education form
df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

#education status
def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == 'Student (Specialist)' or edu_status == "Student (Bachelor's)" or edu_status == "Student (Master's)":
        return 1
    elif edu_status == "Alumnus (Bachelor's)" or edu_status == "Alumnus (Master's)" or edu_status == 'Alumnus (Specialist)':
        return 2
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)

#langs

def Langs_apply(langs):
    if langs.find('Русский') != -1 and langs.find('English') != -1:
        return 0
    else:
        return 1
df['langs'] = df['langs'].apply(Langs_apply)

#occupation type

df['occupation_type'].fillna('university',inplace = True)
def occupation_type_apply(ocu_type):
    if ocu_type == 'university':
        return 0
    return 1
df['occupation_type'] = df['occupation_type'].apply(occupation_type_apply)

#модель

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('   result',axis = 1)
y = df['   result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.40)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred) * 100, 2))