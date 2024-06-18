import sys
import os
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
import re
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn import datasets, linear_model
import pandas as pd 
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import SwarmPackagePy
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from BAT import BAT
from SwarmPackagePy import testFunctions as tf
from BEE import BEE


main = tkinter.Tk()
main.title("Heart Disease Prediction Using Bio Inspired Algorithms")
main.geometry("1300x1200")

global filename
global train
global ga_acc, bat_acc, bee_acc
global classifier


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="heart_dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")


def prediction(X_test, cls):  #prediction done here
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred


# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred, details):
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) * 100
    update_text_widget(details + "\n\n")
    update_text_widget("Accuracy : " + str(accuracy) + "\n\n")
    update_text_widget("Report : " + str(classification_report(y_test, y_pred)) + "\n")
    update_text_widget("Confusion Matrix : " + str(cm) + "\n\n\n\n\n")
    return accuracy


def geneticAlgorithm():
    global classifier
    clear_text_widget()
    global ga_acc
    train = pd.read_csv(filename)
    test = pd.read_csv('heart_dataset/test.txt')
    test_X = test.values[:, 0:12]
    X = train.values[:, 0:12]
    y = train.values[:, 13]

    estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")

    selector = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=0,
                                  scoring="accuracy",
                                  max_features=10,
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=200,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  n_gen_no_change=10,
                                  caching=True,
                                  n_jobs=-1)

    # Perform the selection fit and prediction without threading
    selector = selector.fit(X, y)
    y_pred = selector.predict(test_X)
    prediction_data = prediction(test_X, selector)
    ga_acc = cal_accuracy(prediction_data, prediction_data, 'GA Algorithm Accuracy, Classification Report & Confusion Matrix')
    classifier = selector


def runBat():
    clear_text_widget()
    global bat_acc
    train = pd.read_csv(filename)
    alh = BAT(train.values, tf.easom_function, -10, 10, 2, 20)
    data = alh.get_agents()
    X = []
    Y = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            X.append(data[i][j][0:13])
            Y.append(data[i][j][13])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    cls = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=0, class_weight='balanced')
    cls.fit(X_train, y_train)
    prediction_data = prediction(X_test, cls)
    bat_acc = cal_accuracy(y_test, prediction_data, 'BAT Algorithm Accuracy, Classification Report & Confusion Matrix')


def runBee():
    clear_text_widget()
    global bee_acc
    train = pd.read_csv(filename)
    alh = BEE(train.values, tf.easom_function, -10, 10, 2, 20)
    data = alh.get_agents()
    X = []
    Y = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            X.append(data[i][j][0:13])
            Y.append(data[i][j][13])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    cls = RandomForestClassifier(n_estimators=30, max_depth=2, random_state=0, class_weight='balanced')
    cls.fit(X_train, y_train)
    prediction_data = prediction(X_test, cls)
    bee_acc = cal_accuracy(y_test, prediction_data, 'BEE Algorithm Accuracy, Classification Report & Confusion Matrix')


def predict():
    clear_text_widget()
    filename = filedialog.askopenfilename(initialdir="dataset")
    if not filename:
        update_text_widget("No file selected.\n")
        return
    test = pd.read_csv(filename)
    test = test.values[:, 0:12]
    total = len(test)
    update_text_widget(filename + " test file loaded\n")
    y_pred = classifier.predict(test)
    for i in range(len(test)):
        print(str(y_pred[i]))
        if str(y_pred[i]) == '0.0':
            update_text_widget("X=%s, Predicted = %s" % (test[i], 'No disease detected') + "\n\n")
        elif str(y_pred[i]) == '1.0':
            update_text_widget("X=%s, Predicted = %s" % (test[i], 'Stage 1 Disease Detected') + "\n\n")
        elif str(y_pred[i]) == '2.0':
            update_text_widget("X=%s, Predicted = %s" % (test[i], 'Stage 2 Disease Detected') + "\n\n")
        elif str(y_pred[i]) == '3.0':
            update_text_widget("X=%s, Predicted = %s" % (test[i], 'Stage 3 Disease Detected') + "\n\n")
        elif str(y_pred[i]) == '4.0':
            update_text_widget("X=%s, Predicted = %s" % (test[i], 'Stage 4 Disease Detected') + "\n\n")



def graph():
    height = [ga_acc, bat_acc, bee_acc]
    bars = ('Genetic Algorithm', 'Bat Algorithm', 'Bee Algorithm')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()


def exit():
    main.destroy()


def update_text_widget(text_to_insert):
    text.insert(END, text_to_insert)


def clear_text_widget():
    text.delete('1.0', END)


font = ('times', 16, 'bold')
title = Label(main, text='Heart Disease Prediction Using Bio Inspired Algorithms')
title.config(bg='brown', fg='white')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 14, 'bold')
uploadButton = Button(main, text="Upload Dataset", command=upload)
uploadButton.place(x=50, y=100)
uploadButton.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=460, y=100)

geneticButton = Button(main, text="Run Genetic Algorithm", command=geneticAlgorithm)
geneticButton.place(x=50, y=150)
geneticButton.config(font=font1)

batButton = Button(main, text="Run BAT Algorithm", command=runBat)
batButton.place(x=330, y=150)
batButton.config(font=font1)

beeButton = Button(main, text="Run BEE Algorithm", command=runBee)
beeButton.place(x=620, y=150)
beeButton.config(font=font1)

predictButton = Button(main, text="Upload & Predict Test Data", command=predict)
predictButton.place(x=850, y=150)
predictButton.config(font=font1)

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=50, y=200)
graphButton.config(font=font1)

exitButton = Button(main, text="Exit", command=exit)
exitButton.place(x=330, y=200)
exitButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10, y=250)
text.config(font=font1)

main.config(bg='brown')
main.mainloop()
