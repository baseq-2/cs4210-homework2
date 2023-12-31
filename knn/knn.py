#-------------------------------------------------------------------------
# AUTHOR: Gabriel Fok
# FILENAME: knn.py
# SPECIFICATION: load in data from training data, then perform 1NN LOO-CV and print error rate
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

classification = {'-': 1, '+': 2}

db = []

#reading the data in a csv file
with open('binary_points.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

#loop your data to allow each instance to be your test set
errors = 0
for k in db:

    #add the training features to the 2D array X removing the instance that will be used for testing in this iteration. For instance, X = [[1, 3], [2, 1,], ...]]. Convert each feature value to
    # float to avoid warning messages
    #--> add your Python code here
    # X =
    X = [[float(point[0]), float(point[1])] for point in db if point != k]

    #transform the original training classes to numbers and add to the vector Y removing the instance that will be used for testing in this iteration. For instance, Y = [1, 2, ,...]. Convert each
    #  feature value to float to avoid warning messages
    #--> add your Python code here
    # Y =
    Y = [float(classification[point[-1]]) for point in db if point != k]

    #store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    #testSample =
    testSample = [[float(k[0]), float(k[1])]]

    #fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict(testSample)[0]

    #compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != float(classification[k[-1]]):
        errors += 1

#print the error rate
#--> add your Python code here
print("Error Rate: ", errors/len(db))
