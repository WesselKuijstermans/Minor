##### this code cannot be run as a jupyter notebook
##### save this code to a file and run it using "anaconda prompt> streamlit run streamlit_demo.py"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
import streamlit as st

@st.cache_resource  # this function is executed for the 1st user request; for subsequent user requests, the cached function result is used
# please note that the code below is identical to the code of hands-on 4
def train_model():
    print("function train_model() is called (to verify the caching behavior of Streamlit)")

    # read the training set
    data = pd.read_csv("ex1data1.txt", header=None)  # read from dataset into Pandas DataFrame variable
    data.head()  # view first few rows of the data
    X = data.iloc[:, 0]  # read first column; upper case for matrix
    y = data.iloc[:, 1]  # read second column; lower case for vector
    m = len(y)  # number of training samples; lower case for scalar
    X = np.array(X).reshape(-1, 1)  # transform to format that sklearn expects

    # perform the regression
    regr = linear_model.LinearRegression()
    regr.fit(X, y)

    # the theta's
    print('theta_0', regr.intercept_, 'theta_1', regr.coef_)
    
    return regr


st.title('Predicting profit as function of city size')

# train the model. This is done only once due to the Streamlit caching feature
regr = train_model()

# perform prediction
city_size = float(st.text_input("Enter the city size: ", '0'))  # 2nd argument is the value of city_size on 1st rendering of the web app
pred = regr.predict([[city_size]])
print(pred)
st.text('The predicted profit for city size ' + str (city_size * 10000) + ' is ' + str(pred * 10000) + ' dollars.') 
