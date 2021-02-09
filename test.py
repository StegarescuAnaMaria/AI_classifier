import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn


#citesc predictia
file=open("C:/Users/ayami/PycharmProjects/Proiect/rezultat.txt", "r")
rezultat=[]
data1=[]
for line in file:
    data1=line.split(',')
    rezultat.append(int(data1[1]))
file.close()
#citesc etichetele
file4=open("C:/Users/ayami/PycharmProjects/Proiect/data2/validation_labels.txt", "r", encoding="utf-8")
data = []
id = ''
clasa = ''
labels=[]
for lines in file4:
    data=lines.split()
    id = data[0]
    clasa = data[1]
    labels.append(int(clasa))
file4.close()

#calculez acuratetea
acuratete=f1_score(labels, rezultat, average='macro')
print(acuratete)

#matricea de confuzie
matrix=confusion_matrix(labels, rezultat)
print(matrix)
plt.figure(figsize = (2,2))
sn.heatmap(matrix, annot=True)

plt.imshow(matrix)
plt.show()