#-------------------------------------------------------------------------
# AUTHOR: Gabriel Fok
# FILENAME: decision_tree_2.py
# SPECIFICATION: using input CSV files, create a decision tree based on the given data. assess accuracy of the model based on the test data
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

ageDict = {"young": 1, "prepresbyopic": 2, "presbyopic": 3}
spectacleDict = {"myope": 1, "hypermetrope": 2} 
astigmatismDict = {"yes": 1, "no": 2} 
tearProductionRateDict = {"reduced": 1, "normal": 2}
classification = {"no": 1, "yes": 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append (row)

    #transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    #--> add your Python code here
    # X =
    X = [[ageDict[row[0].lower()], spectacleDict[row[1].lower()], astigmatismDict[row[2].lower()], tearProductionRateDict[row[3].lower()]] for row in dbTraining]

    #transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =
    Y = [classification[row[-1].lower()] for row in dbTraining]

    #loop your training and test tasks 10 times here
    accuracy = 0
    for i in range (10):
        #fitting the decision tree to the data setting max_depth=3
        clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
        clf = clf.fit(X, Y)

        #read the test data and add this data to dbTest
        #--> add your Python code here
        # dbTest =
        dbTest = []
        with open('contact_lens_test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0: #skipping the header
                    dbTest.append (row)
        
        for data in dbTest:
            #transform the features of the test instances to numbers following the same strategy done during training,
            #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
            #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            #--> add your Python code here
            A = [ageDict[data[0].lower()], spectacleDict[data[1].lower()], astigmatismDict[data[2].lower()], tearProductionRateDict[data[3].lower()]]
            class_predicted = clf.predict([A])[0]

            #compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            #--> add your Python code here
            if class_predicted == classification[data[4].lower()]:
                accuracy = accuracy + 1

    #find the average of this model during the 10 runs (training and test set)
    #--> add your Python code here
    accuracy = accuracy / (10 * len(dbTest))

    #print the average accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    #--> add your Python code here
    print(f"final accuracy when training on {ds}: {accuracy}")

