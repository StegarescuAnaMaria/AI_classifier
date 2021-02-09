import numpy as np
import collections
from sklearn import preprocessing
from sklearn import discriminant_analysis
import sklearn
import math
from stop_words import get_stop_words
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC



class BagOfWords:
    def __init__(self):
        self.dict={}
        self.list_words=[]
        self.list_sentences=[]
        self.labels=[]
        self.allwords=[]
        self.id=[]
        #citirea datelor
    def build_vocabulary(self, file):
        for line in file:
            line1 = line.lower()
            data = line1.split()
            id = data[0]
            self.id.append(id)
            data.remove(id)
            #lista cu fragmentele de text
            self.list_sentences.append(data)
            for word in data:
                #lista cu toate cuvintele
                    self.allwords.append(word)
                    if(not word in self.dict):
                        self.dict[word]=id
                        #lista de cuvinte distincte
                        self.list_words.append(word)
        file.close()

#metoda citeste etichetele si le salveaza intr-o lista
    def readLabels(self, file):
        data = []
        id = ''
        clasa = ''
        for lines in file:
            data=lines.split()
            id = data[0]
            clasa = data[1]
            self.labels.append(clasa)
        file.close()

#metoda construieste matricea de caracteristici
    def get_features(self):
        features=np.zeros((len(self.list_sentences), len(self.list_words)))
        for i in range(len(self.list_sentences)):
            #Counter returneaza un dictionar cu cheile - valori din lista initiala,
            # si valori - numarul de repetitii a cheii date in lista
            d = collections.Counter(self.list_sentences[i])
            for j in range(len(self.list_words)):
                if(self.list_words[j] in d):
                    features[i][j]=d[self.list_words[j]]
                else:
                    features[i][j]=0
        return features

#metoda pastreaza cele mai frecvente n_features cuvinte
    def sortWords(self, n_features):
        d=collections.Counter(self.allwords)
        #most common returneaza cuvintele cele mai frecvente
        d2=d.most_common(n_features)
        a=[]
        for item, nr in d2:
            a.append(item)
        self.list_words=a

#normalizeaza matricea de features si o returneaza
def normalize_data(train_data, type=None):
    if (type=='standard'):
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        scaled_train = scaler.transform(train_data)
        return scaled_train
    elif(type=='l2'):
        norma=0
        train_scaled=[]
        for i in range(len(train_data)):
            for j in range(len(train_data[i])):
                norma+=train_data[i][j]**2
            norma=math.sqrt(norma)
            train_scaled.append(train_data[i]/norma)
            norma=0
        print(train_scaled)


file=open("C:/Users/ayami/PycharmProjects/Proiect/data2/train_samples.txt", "r", encoding="utf-8")
file2=open("C:/Users/ayami/PycharmProjects/Proiect/data2/train_labels.txt", "r", encoding="utf-8")
trainObj=BagOfWords()
trainObj.readLabels(file2)
trainObj.build_vocabulary(file)
file3=open("C:/Users/ayami/PycharmProjects/Proiect/data2/validation_samples.txt", "r", encoding="utf-8")
testObj=BagOfWords()
testObj.build_vocabulary(file3)
#Ma asigur ca matricile de features sa aiba un numar de coloane egale (ie un numar egal de cuvinte pentru
# datele de train si test)
if(len(trainObj.list_words)>len(testObj.list_words)):
    trainObj.sortWords(len(testObj.list_words))
else:
    testObj.sortWords(len(trainObj.list_words))
train_features=trainObj.get_features()
test_features=testObj.get_features()
normalized_train=normalize_data(train_features, 'standard')
normalized_test=normalize_data(test_features, 'standard')

#Incercari pe diversi clasificatori

#svc=SGDClassifier(loss='squared_epsilon_insensitive', max_iter=1200, shuffle=True)
#svc=svm.SVC(C=1.0, kernel='linear', gamma='auto')
#svc=SGDClassifier(loss='squared_epsilon_insensitive', l1_ratio=0.25, fit_intercept=True, max_iter=2000, shuffle=True, tol=0.0001, verbose=2,
 #              n_iter_no_change=10, epsilon=0.01, n_jobs=-1)
#svc=discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr')
#svc=LinearSVC()
#svc=GaussianNB()
svc = MLPClassifier(activation='logistic', hidden_layer_sizes=110)
#y_pred = gnb.fit(normalized_train, trainObj.labels).predict(normalized_test)

#kernel = DotProduct() + WhiteKernel()
#gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(normalized_train, trainObj.labels)
#rezultat=gpr.predict(test_features)
svc.fit(normalized_train, trainObj.labels)
rezultat=svc.predict(normalized_test)
file_rez=open("rezultat.txt", "w")
#file_rez.write("id,label\n")
for i in range(len(testObj.id)):
    file_rez.write(str(testObj.id[i])+","+rezultat[i]+"\n")

