# Import the libraries

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from mlxtend.plotting import plot_decision_regions

# load the dataset
df = sns.load_dataset('iris')

# make X and y
X = df.drop(columns=df.columns[2:])
y = df['species']

# encode the target variable

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(y)

# instantiate the models

from sklearn.ensemble import VotingClassifier

lr = LogisticRegression()
dt = DecisionTreeClassifier()
svm = SVC(probability=True)

import streamlit as st

# set the title for the app
st.title('Voting Classifier')

# make the options sidebar for different models in a multi-select
st.sidebar.title('Options')

model_select = st.sidebar.multiselect(label='Select the Classifier',
                                      options=['Logistic regression', 'Decision Trees', 'SVM']
                                    )
estimator_list = []

for model in model_select:
    if model == 'Logistic regression':
        estimator_list.append(('Logistic Regression', lr))
    elif model == 'Decision Trees':
        estimator_list.append(('Decision Trees', dt))
    elif model == 'SVM':
        estimator_list.append(('SVM', svm))

# option to toggle decision boundaries

select_button = st.sidebar.checkbox('Select to view Decision Boundaries', value=False,key=1)

if select_button:
    st.subheader('Decision Boundaries of Classifiers on the dataset')
    column_num = len(estimator_list)

    if column_num == 1:
        model = estimator_list[0][1]
        model.fit(X, y)
        y_pred = model.predict(X)
        score = cross_val_score(model, X, y, cv=5, scoring='accuracy')

        fig, ax = plt.subplots()
        plot_decision_regions(X.values, y, model)
        st.pyplot(fig)

        st.write(f'The score of {estimator_list[0][0]} is', np.round(np.mean(score),2))

    if column_num == 2:
        col1, col2 = st.columns(2)

        with col1:
            model1 = estimator_list[0][1]
            model1.fit(X, y)
            y_pred1 = model1.predict(X)
            score = cross_val_score(model1, X, y, cv=5, scoring='accuracy')

            fig1, ax1 = plt.subplots()
            plot_decision_regions(X.values, y, model1)
            st.pyplot(fig1)

            st.write(f'The score of {estimator_list[0][0]} is', np.round(np.mean(score),2))

        with col2:
            model2 = estimator_list[1][1]
            model2.fit(X, y)
            y_pred2 = model2.predict(X)
            score = cross_val_score(model2, X, y, cv=5, scoring='accuracy')

            fig2, ax2 = plt.subplots()
            plot_decision_regions(X.values, y, model2)
            st.pyplot(fig2)

            st.write(f'The score of {estimator_list[1][0]} is', np.round(np.mean(score),2))

    if column_num == 3:
        col1, col2, col3 = st.columns(3)

        with col1:
            model1 = estimator_list[0][1]
            model1.fit(X, y)
            y_pred1 = model1.predict(X)
            score = cross_val_score(model1, X, y, cv=5, scoring='accuracy')

            fig1, ax1 = plt.subplots()
            plot_decision_regions(X.values, y, model1)
            st.pyplot(fig1)

            st.write(f'The score of {estimator_list[0][0]} is', np.round(np.mean(score),2))

        with col2:
            model2 = estimator_list[1][1]
            model2.fit(X, y)
            y_pred2 = model2.predict(X)
            score = cross_val_score(model2, X, y, cv=5, scoring='accuracy')

            fig2, ax2 = plt.subplots()
            plot_decision_regions(X.values, y, model2)
            st.pyplot(fig2)

            st.write(f'The score of {estimator_list[1][0]} is', np.round(np.mean(score),2))

        with col3:
            model3 = estimator_list[2][1]
            model3.fit(X, y)
            y_pred3 = model3.predict(X)
            score = cross_val_score(model3, X, y, cv=5, scoring='accuracy')

            fig3, ax3 = plt.subplots()
            plot_decision_regions(X.values, y, model3)
            st.pyplot(fig3)

            st.write(f'The score of {estimator_list[2][0]} is', np.round(np.mean(score),2))


# set the weights of the classifiers

if len(estimator_list) > 0:
    weight_checkbox = st.sidebar.checkbox(label='Toggle Classifier weights', disabled=False, key=2)
else:
    weight_checkbox = st.sidebar.checkbox(label='Toggle Classifier weights', disabled=True, key=2)

if weight_checkbox:
    if len(estimator_list) == 1:
        weight_list = []
        select1 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[0][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)
        weight_list.append(select1)
        st.sidebar.write(f'The weight of {estimator_list[0][0]} classifier is {weight_list[0]}')

    elif len(estimator_list) == 2:
        weight_list = []
        select1 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[0][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)
        select2 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[1][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)
        weight_list.append(select1)
        weight_list.append(select2)
        st.sidebar.write(f'The weight of {estimator_list[0][0]} classifier is {weight_list[0]}')
        st.sidebar.write(f'The weight of {estimator_list[1][0]} classifier is {weight_list[1]}')

    elif len(estimator_list) == 3:
        weight_list = []

        select1 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[0][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)
        select2 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[1][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)
        select3 = st.sidebar.selectbox(label=f'Select the Weights of {estimator_list[2][0]} classifier',
                                       options=[1, 2, 3, 4, 5],
                                       index=0)

        weight_list.append(select1)
        weight_list.append(select2)
        weight_list.append(select3)
        st.sidebar.write(f'The weight of {estimator_list[0][0]} classifier is {weight_list[0]}' )
        st.sidebar.write(f'The weight of {estimator_list[1][0]} classifier is {weight_list[1]}')
        st.sidebar.write(f'The weight of {estimator_list[2][0]} classifier is {weight_list[2]}')


# select the type of voting classifier (Hard/Soft)

radio_button = st.sidebar.radio(label='Select the Type of Voting Classifier', options=['Soft', 'Hard'], index=1)
type_button = st.sidebar.button('Confirm Selection')

if weight_checkbox:

    if radio_button == 'Soft' and type_button:
        st.subheader('Decision Boundary of Voting Classifier using `Soft` voting ')
        clf = VotingClassifier(estimators=estimator_list, voting='soft',weights=weight_list)
        clf.fit(X, y)

        score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

        fig, ax = plt.subplots()
        plot_decision_regions(X.values, y, clf)
        st.pyplot(fig)

        st.write('The mean score of Voting Classifier is', np.round(score.mean(),2))

    elif radio_button == 'Hard' and type_button:
        st.subheader('Decision Boundary of Voting Classifier using `Hard` voting ')
        clf = VotingClassifier(estimators=estimator_list, voting='hard',weights=weight_list)
        clf.fit(X, y)
        score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

        fig, ax = plt.subplots()
        plot_decision_regions(X.values, y, clf)
        st.pyplot(fig)

        st.write('The mean score of Voting Classifier is', np.round(score.mean(),2))

else:

    if radio_button == 'Soft' and type_button:
        st.subheader('Decision Boundary of Voting Classifier using `Soft` voting ')
        clf = VotingClassifier(estimators=estimator_list, voting='soft')
        clf.fit(X, y)

        score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

        fig, ax = plt.subplots()
        plot_decision_regions(X.values, y, clf)
        st.pyplot(fig)

        st.write('The mean score of Voting Classifier is', np.round(score.mean(), 2))

    elif radio_button == 'Hard' and type_button:
        st.subheader('Decision Boundary of Voting Classifier using `Hard` voting ')
        clf = VotingClassifier(estimators=estimator_list, voting='hard')
        clf.fit(X, y)
        score = cross_val_score(clf, X, y, cv=10, scoring='accuracy')

        fig, ax = plt.subplots()
        plot_decision_regions(X.values, y, clf)
        st.pyplot(fig)

        st.write('The mean score of Voting Classifier is', np.round(score.mean(), 2))
