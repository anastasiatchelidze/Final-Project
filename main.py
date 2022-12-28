import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from PyQt5 import QtWidgets
import sys
from pymongo import MongoClient
from qtgui import Ui_MainWindow
import matplotlib.pyplot as plt
from threading import Thread


class Health:
    """
       A class used to model person's health data

       ...

       Attributes
       ----------
       sgr : float
           represents fasting blood sugar level
       hem : float
           represents hemoglobin level

       Methods
       -------
        __init__(self, sgr, hem):
           Initializes attributes with passed arguments
        __lt__(self, other):
           Overrides less than operator by comparing hemoglobin levels
        __str__(self):
            Overrides __str__ method, allows printing info about Health object
       """

    def __init__(self, sgr, hem):
        self.sgr = sgr
        self.hem = hem

    def __lt__(self, other):
        """
        Overrides less than operator

            Parameters:
                    other (Health): Another Health object for comparison

            Returns:
                    bool: True if first Health object's hemoglobin is less than second and False otherwise
        """
        return self.hem < other.hem

    def __str__(self):
        """
        Overrides __str__ method, prints blood sugar level and hemoglobin

            Returns:
                    formatted string (str): attribute names with their amount
        """
        return f"Blood Sugar level: {self.sgr}\nHemoglobin: {self.hem}\n"


class Person(Health):
    """
           A class used to model person (inherits from class Health)

       ...

       Attributes
       ----------
        gnd : char
           represents person's gender
        age : int
           indicates person's age
        ht : int
           represents person's height in cm
        wt : float
           indicates person's weight in kg
        sgr : float
           represents fasting blood sugar level
        hem : float
           represents hemoglobin level
        smoking : int
           denotes a non-smoker with 0 and smoker with 1

       Methods
       -------
        __init__(self, gnd, age, ht, wt, sgr, hem, smoking):
           Initializes attributes with passed arguments
        is_smoker(self):
            Returns boolean indicating whether a person smokes or not
        __str__(self):
            Overrides __str__ method, allows printing info about Person object
           """

    def __init__(self, gnd, age, ht, wt, sgr, hem, smoking):
        super().__init__(sgr, hem)
        self.gnd = gnd
        self.age = age
        self.ht = ht
        self.wt = wt
        self.smoking = smoking

    def is_smoker(self):
        """
        Returns boolean (True or False) to indicate whether a person is a smoker or non-smoker

            Returns:
                    bool: True if person smokes and False otherwise
        """
        if self.smoking == 1:
            return True
        else:
            return False

    def __str__(self):
        """
        Overrides __str__ method, prints person's gender, age and whether he/she smokes

            Returns:
                    formatted string (str): attribute names with their amount, uses is_smoker() method
        """
        return f"Gender: {self.gnd}\nAge: {self.age}\nIs smoker: {self.is_smoker()}"


class MyApp(QtWidgets.QMainWindow):
    """
           A class used to represent GUI application, inherits from class QMainWindow of QtWidgets

           ...

           Attributes
           ----------
            ui : str
               a formatted string to print out what the animal says
            conn : str
               the name of the animal
            db : str
               the sound that the animal makes
            coll : int
               the number of legs the animal has (default 4)

           Methods
           -------
            calc_bmi(self):
               Calculates and displays person's BMI
            add_record(self, pers: Person)
                adds new row of information using Person object and its data
            delete_record(self, pers: Person):
                deletes row of information using Person object and its data
            show_graphs(self):
                displays two charts in different windows,
                a pie chart showing the percentage of smokers and
                ROC curve of logistic regression model
           """

    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # creating collection inside a database in MongoDB
        self.conn = MongoClient()
        self.db = self.conn["database"]
        self.coll = self.db["smoking_data"]

        # connecting functionality (calculating BMI) to the button
        self.ui.calcButton.clicked.connect(self.calc_bmi)

    def calc_bmi(self):
        """
            Calculates person's BMI (Body Mass Index)

            Raises:
                ZeroDivisionError
                    if height (denominator) is zero, error is raised and outputted.
            """
        if self.ui.dispbmi.toPlainText() != "":
            self.ui.dispbmi.setText("")
        try:
            h = self.ui.inputh.toPlainText()
            w = self.ui.inputw.toPlainText()
            result = float(w) / (float(h) / 100) ** 2
            self.ui.dispbmi.setText(f"{result}")

            # clearing input fields
            self.ui.inputh.clear()
            self.ui.inputw.clear()
        except ZeroDivisionError:
            self.ui.dispbmi.setText("Error.")

    def add_record(self, pers: Person):
        """
            Adds new record - row of data with Person attributes

                Parameters:
                    pers (Person): Person object for inserting info in collection
                """
        temp_dct = {
            "gender": pers.gnd,
            "age": pers.age,
            "height": pers.ht,
            "weight": pers.wt,
            "blood sugar": pers.sgr,
            "hemoglobin": pers.hem,
            "smoking": pers.smoking
        }

        self.coll.insert_one(temp_dct)

    def delete_record(self, pers: Person):
        """
            Deletes record - row of data with Person attributes

                Parameters:
                    pers (Person): Person object for deleting info from the collection
                """
        self.coll.delete_one({
            "gender": pers.gnd,
            "age": pers.age,
            "height": pers.ht,
            "weight": pers.wt,
            "blood sugar": pers.sgr,
            "hemoglobin": pers.hem,
            "smoking": pers.smoking
        })

    def show_graphs(self):
        """
            Shows pie chart for comparison of the number of smokers and non-smokers and ROC curve for evaluating
            the logistic regression model of how blood sugar level and hemoglobin are correlated with smoking
            """
        # downloading and processing the data
        self.df = pd.read_csv("smoking.csv")
        self.df.drop(columns=['hearing(left)', 'hearing(right)', 'triglyceride', 'HDL', 'LDL', 'Urine protein',
                              'serum creatinine', 'AST', 'ALT', 'Gtp', 'oral', 'dental caries', 'tartar',
                              'eyesight(left)', 'eyesight(right)', 'relaxation', 'systolic', 'ID', 'waist(cm)',
                              'Cholesterol'], inplace=True)
        self.df = self.df.reset_index()  # resetting indexing after cleaning

        # counting the number of smokers to obtain their percentage
        df1 = self.df['smoking'].groupby(self.df['smoking']).count()
        a = plt.figure(1)
        plt.pie(df1, labels=["Non-smokers", "Smokers"], autopct='%1.0f%%')
        a.legend(["Non-smokers", "Smokers"])
        a.show()

        features = ['fasting blood sugar', 'hemoglobin']  # features of logistic regression model
        X = np.array(self.df[features])
        y = self.df.smoking  # initializing target variable

        # Splitting data into train data and test data to ensure stability of the model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=7)
        logreg = LogisticRegression(max_iter=200, random_state=16)  # creating the model
        logreg.fit(X_train, y_train)  # fitting the model
        y_pred = logreg.predict(X_test)

        target_names = ['Not Smoking', 'Smoking']
        # evaluating the model by comparing test data and predicted data with classification report
        print(classification_report(y_test, y_pred, target_names=target_names))

        y_pred_proba = logreg.predict_proba(X_test)[::, 1]
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)  # FP (false positive rate) and TP (true positive rate)
        auc = metrics.roc_auc_score(y_test, y_pred_proba)  # visualizing how well the model works
        b = plt.figure(2)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        b.show()


# global variable - number of young adults in the data
ya_count = 0


def is_young(row):
    """
        Counts the number of young adults of the data file

            Parameters:
                    row (str): a row of data read from the file
        """
    row_data = row.split(',')
    if int(row_data[2]) <= 30:
        global ya_count
        ya_count += 1


def thr_func(funcName, fileName):
    """
        Runs function (passed as an argument) in several threads.

            Parameters:
                    funcName (function): name of the function
                    fileName (str): string of name of the file

            Raises FileNotFoundError:
                If no file with specified name is found, error is raised and correspondingly outputted.
        """
    try:
        with open(fileName, 'r') as file:
            arr = file.readlines()
            arr.pop(0)  # removing the header row from the array
        th_arr = []
        for row in arr:
            t = Thread(target=funcName, args=(row,))
            th_arr.append(t)

        for th in th_arr:
            th.start()

        for th in th_arr:
            th.join()
    except FileNotFoundError:
        print(f"File with name '{fileName}' does not exist.")
    finally:
        return f"Young adult count: {ya_count}\n"


print(thr_func(is_young, "smoking.csv"))

app = QtWidgets.QApplication([])
application = MyApp()
application.show_graphs()
application.show()
sys.exit(app.exec())