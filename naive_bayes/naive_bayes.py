#-------------------------------------------------------------------------
# AUTHOR: Gabriel Fok
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#reading the training data in a csv file
#--> add your Python code here
db = []
outlookDict = {"sunny": 1, "overcast": 2, "rain": 3}
temperatureDict = {"hot": 1, "mild": 2, "cool": 3}
humidityDict = {"high": 1, "normal": 2}
windyDict = {"weak": 1, "strong": 2}
classification = {"no": 1, "yes": 2}

with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)


#transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
# X =
X = [[outlookDict[row[1].lower()], temperatureDict[row[2].lower()], humidityDict[row[3].lower()], windyDict[row[4].lower()]] for row in db]

#transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
# Y =
Y = [classification[row[-1].lower()] for row in db]

#fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

#reading the test data in a csv file
#--> add your Python code here
dbTest = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            dbTest.append (row)

#printing the header os the solution
#--> add your Python code here
print(f"{'Day':<5}{'Outlook':<10}{'Temperature':<15}{'Humidity':<15}{'Windy':<10}{'PlayTennis':<12}{'Confidence':<10}")

#use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for test in dbTest:
    probabilities = list(clf.predict_proba([[outlookDict[test[1].lower()], temperatureDict[test[2].lower()], humidityDict[test[3].lower()], windyDict[test[4].lower()]]])[0])
    if max(probabilities) > 0.75:
        if max(probabilities) == probabilities[0]:
            print(f"{test[0]:<5}{test[1]:<10}{test[2]:<15}{test[3]:<15}{test[4]:<10}{'No':<12}{max(probabilities):.2f}")
        else:
            print(f"{test[0]:<5}{test[1]:<10}{test[2]:<15}{test[3]:<15}{test[4]:<10}{'Yes':<12}{max(probabilities):.2f}")

