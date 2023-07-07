#Checkpoint Objective
#Build a Streamlit app that predicts the type of iris flower based on user input using a Random Forest Classifier


#Import the necessary libraries: Streamlit, sklearn.datasets, and sklearn.ensemble
import streamlit as st
from sklearn.datasets import load_iris
import sklearn.ensemble
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics


#Load the iris dataset using the "datasets.load_iris()" function and assign the data and target variables to "X" and "Y", respectively.
data = load_iris()
df = pd.DataFrame(data=data.data, columns=data.feature_names)
df.head()
X = df[data.feature_names]
targ = pd.Series(data.target)
Y = targ

#Set up a Random Forest Classifier and fit the model using the "RandomForestClassifier()" and "fit()" functions.
#Apply random forest
#Use random forest then change the number of estimators


# Apply one-hot encoding to categorical columns
x_encoded = pd.get_dummies(X)

# Splitting data

x_train, x_test, y_train, y_test = train_test_split(x_encoded, Y, test_size=0.20, random_state=10)

#Random Forest prediction

RFC=RandomForestClassifier(n_estimators=10)  #Creating a random forest with 10 decision trees
RFC.fit(x_train, y_train)  #Training our model
y_pred=RFC.predict(x_test)  #testing our model
Accuracy_RandForest = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:", Accuracy_RandForest)  #Measuring the accuracy of our model

#Create a Streamlit app using the "streamlit.title()" and "streamlit.header()" functions to add a title and header to the app.

#Title
st.title("Iris Dataset")

#Add target series to our DataFrame
df = pd.DataFrame(data=data.data, columns=data.feature_names)
ser = pd.Series(data.target)
new_hr = df.assign(Target=ser)

#DataFrame
st.dataframe(new_hr)
st.write("0 : setosa  🌸  1 : versicolor 🌼  3 : virginica")
st.divider()

#Header
st.header("Random Forest Classifier 🌈")


#Add input fields for sepal length, sepal width, petal length, and petal width using the "streamlit.slider()" function.
# Use the minimum, maximum, and mean values of each feature as the arguments for the function.

a = df['sepal length (cm)'].mean()
b = df['sepal length (cm)'].max()
c = df['sepal length (cm)'].min()

d = df['sepal width (cm)'].mean()
e = df['sepal width (cm)'].max()
f = df['sepal width (cm)'].min()

g = df['petal length (cm)'].mean()
h = df['petal length (cm)'].max()
i = df['petal length (cm)'].min()

j = df['petal width (cm)'].mean()
k = df['petal width (cm)'].max()
l = df['petal width (cm)'].min()


level_1 = st.slider("Select the level", c, a, b)
level_2 = st.slider("Select the level", f, d, e)
level_3 = st.slider("Select the level", i, g, h)
level_4 = st.slider("Select the level", l, j, k)
print(level_1)
print(level_2)
print(level_3)
print(level_4)

# print the level
st.text(f'Selected: {level_1}'.format(level_1))
st.text(f'Selected: {level_2}'.format(level_2))
st.text(f'Selected: {level_3}'.format(level_3))
st.text(f'Selected: {level_4}'.format(level_4))


#Define a prediction button using the "streamlit.button()" function that takes in the input values and uses
# the classifier to predict the type of iris flower.

# Define a prediction button
#if st.button('Prediction'):
    # Make the prediction
    #features = df[data.feature_names]
    #features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Display the predicted type of iris flower
    #st.write(f'Predicted Iris Type: {targ[y_pred[0]]}')


# Define a prediction button
if st.button('Predict'):
    # Make the prediction
    features = df[data.feature_names]
    prediction = RFC.predict(features)

    # print the BMI INDEX
    st.text("Your type is {}.".format(y_pred[0]))


