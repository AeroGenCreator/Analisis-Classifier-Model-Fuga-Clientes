import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, confusion_matrix

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from plotly import express as px
from plotly import graph_objects as go

st.set_page_config(layout='wide')

SEED = 12345

@st.cache_data
def load_data():
    df = pd.read_csv('data_limpia.csv')
    return df

@st.cache_data
def get_train_validation_test(data:pd.DataFrame):
    train, temporal = train_test_split(data,random_state=SEED,test_size=0.4)
    validation, test = train_test_split(data,random_state=SEED,test_size=0.5)
    return train, validation, test

def get_features_target(data:pd.DataFrame,target_column):
    X = data.drop(columns=target_column)
    Y = data[target_column]
    return X, Y

def logistic_regresion(train,validation):
    
    solver = 'liblinear'
    class_weight = 'balanced'

    train_features, train_target = get_features_target(train,target_column='Exited')
    validation_features, validation_target = get_features_target(validation,target_column='Exited')

    model = LogisticRegression(random_state=SEED,solver=solver,class_weight=class_weight)
    model.fit(train_features,train_target)

    prediction_validation = model.predict(validation_features)

    probability_validation_class_one = model.predict_proba(validation_features)[:,1]
    false_positive_rate, true_positive_rate, threshold = roc_curve(validation_target,probability_validation_class_one)

    fig = px.line(x=false_positive_rate,y=true_positive_rate,title='ROC Curve LogisticRegression')
    trace_line_dummy_prediction = go.Scatter(x=[0,1], y=[0,1],mode='lines', name='Dummy Prediction',line=dict(color='Red', width=1))
    fig.append_trace(trace_line_dummy_prediction,row=1,col=1)
    fig.update_xaxes(title_text='False Positive Rate',col=1,row=1)
    fig.update_yaxes(title_text='True Positive Rate (Recall)',row=1,col=1)
    st.plotly_chart(fig)

    st.markdown(f'**F1 SCORE:** LogisticRegression: `{f1_score(validation_target,prediction_validation)}`, **Hyperparameters:** Solver: `{solver}`, Class Weight: `{class_weight}`.')
    st.markdown(f'**ROC SCORE:** `{roc_auc_score(validation_target,model.predict_proba(validation_features)[:, 1])}`.')

    return model

def decision_tree_classifier(train,validation):
    
    max_depth = 5
    min_samples_split = 10
    min_samples_leaf = 25
    class_weight = 'balanced'

    train_features, train_target = get_features_target(data=train,target_column='Exited')
    validation_features, validation_target = get_features_target(data=validation,target_column='Exited')

    model = DecisionTreeClassifier(random_state=SEED,max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,class_weight=class_weight)
    model.fit(train_features, train_target)

    prediction_validation = model.predict(validation_features)

    probability_validation_class_one = model.predict_proba(validation_features)[:,1]
    false_positive_rate, true_positive_rate, treshold = roc_curve(validation_target,probability_validation_class_one)

    fig = px.line(x=false_positive_rate,y=true_positive_rate,title='ROC Curve DecisionTreeClassifier')
    trace_line_dummy_prediction = go.Scatter(x=[0,1], y=[0,1],mode='lines', name='Dummy Prediction',line=dict(color='Red', width=1))
    fig.append_trace(trace_line_dummy_prediction,row=1,col=1)
    fig.update_xaxes(title_text='False Positive Rate',col=1,row=1)
    fig.update_yaxes(title_text='True Positive Rate (Recall)',row=1,col=1)
    st.plotly_chart(fig)

    st.markdown(
        f'**F1 SCORE:** DecisionTreeClassifier: `{f1_score(validation_target,prediction_validation)}`, **Hyperparameters:** max_depth: `{max_depth}`, min_samples_split: `{min_samples_split}`, min_samples_leaf: `{min_samples_leaf}`, class_weight: `{class_weight}`.')
    st.markdown(f'**ROC SCORE:** `{roc_auc_score(validation_target,model.predict_proba(validation_features)[:, 1])}`.')
    
    return model

def random_forest_classifier(train,validation):

    n_estimators = 200
    max_depth = 50
    max_leaf_nodes = 25
    min_samples_leaf = 25
    class_weight = 'balanced'

    train_features, train_target = get_features_target(data=train,target_column='Exited')
    validation_features, validation_target = get_features_target(data=validation,target_column='Exited')

    model = RandomForestClassifier(random_state=SEED,n_estimators=n_estimators,max_depth=max_depth,max_leaf_nodes=max_leaf_nodes,min_samples_leaf=min_samples_leaf,class_weight=class_weight)
    model.fit(train_features,train_target)

    prediction_validation = model.predict(validation_features)

    probability_validation_class_one = model.predict_proba(validation_features)[:,1]
    false_positive_rate, true_positive_rate, treshold = roc_curve(validation_target,probability_validation_class_one)

    fig = px.line(x=false_positive_rate,y=true_positive_rate,title='ROC Curve RandomForestClassifier')
    trace_line_dummy_prediction = go.Scatter(x=[0,1], y=[0,1],mode='lines', name='Dummy Prediction',line=dict(color='Red', width=1))
    fig.append_trace(trace_line_dummy_prediction,row=1,col=1)
    fig.update_xaxes(title_text='False Positive Rate',col=1,row=1)
    fig.update_yaxes(title_text='True Positive Rate (Recall)',row=1,col=1)
    st.plotly_chart(fig)

    st.markdown(
        f'**F1 SCORE:** RandomForestClassifier: `{f1_score(validation_target,prediction_validation)}`, **Hyperparameters:** max_depth: `{max_depth}`, n_estimators: `{n_estimators}`, min_samples_leaf: `{min_samples_leaf}`, max_leaf_nodes: `{max_leaf_nodes}`, class_weight: `{class_weight}`.')
    st.markdown(f'**ROC SCORE:** `{roc_auc_score(validation_target,model.predict_proba(validation_features)[:, 1])}`.')
    
    return model

# ***************** INTERFAZ:
st.title(':material/currency_bitcoin: Classifier Model Testing On Finance Data Set')
st.subheader('The objective for Beta Bank is to build a machine learning model that predicts which customers are about to leave. Model Objective: To achieve a minimum F1-score of 0.59.')

df = load_data()
train, validation, test = get_train_validation_test(df)

with st.expander('Data Preview'):
    st.dataframe(df,column_order=['CreditScore','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','Geography_Germany','Geography_Spain','Gender_Male','Exited'])

st.markdown(f'**Class Balance `(0\'s,1\'s)` `{train['Exited'].value_counts(normalize=1).tolist()}`.')
st.divider()

st.header(':material/graph_1: Different Models Tested on Validation Set')
col1, col2, col3 = st.columns(3)

with col1:
    with st.expander('LogisticRegression'):
        logistic_regresion_model = logistic_regresion(train=train,validation=validation)
with col2:
    with st.expander('DecisionTreeClassifier'):
        decision_tree_classifier_model = decision_tree_classifier(train=train,validation=validation)
with col3:
    with st.expander('RandomForestClassifier'):
        random_forest_classifier_model = random_forest_classifier(train=train,validation=validation)

st.divider()
st.header(':material/forest: Testing RandomForestClassifier on Test Set')

test_features, test_target = get_features_target(data=test,target_column='Exited')

test_prediction = random_forest_classifier_model.predict(test_features)
test_class_one_probability = random_forest_classifier_model.predict_proba(test_features)[:,1]
false_positive_rate, true_positive_rate, treshold = roc_curve(test_target,test_class_one_probability)

fig = px.line(x=false_positive_rate,y=true_positive_rate,title='ROC Curve RandomForestClassifier TEST DATA')
trace_line_dummy_prediction = go.Scatter(x=[0,1], y=[0,1],mode='lines', name='Dummy Prediction',line=dict(color='purple', width=1))
fig.append_trace(trace_line_dummy_prediction,row=1,col=1)
fig.update_xaxes(title_text='False Positive Rate',col=1,row=1)
fig.update_yaxes(title_text='True Positive Rate (Recall)',row=1,col=1)

col4, col5, col6 = st.columns(3)
with col4:
    st.plotly_chart(fig)

with col5:
    st.divider()
    st.markdown(f'**F1 SCORE:** RandomForestClassifier: `{f1_score(test_target,test_prediction)}`.')
    st.markdown(f'**ROC SCORE:** `{roc_auc_score(test_target,random_forest_classifier_model.predict_proba(test_features)[:, 1])}`.')

    st.markdown('**CONFUSSION MATRIX**')
    matrix = confusion_matrix(test_target,test_prediction)
    st.dataframe(pd.DataFrame(data=matrix,columns=['Predictio: 0 (No Exited)', 'Prediction: 1 (Exited)'],index=['Real: 0 (No Exited)', 'Real: 1 (Exited)']))
    st.markdown('**:red[CONCLUSION:]** Best Model: RandomForestClassifier. This model reached :red[$60$%] score on test data. Recall for (exited users) detected :red[$755$] out of :red[$1021$] proves the ability of the model for this finance project.')
    st.markdown('However this recall classified :red[$764$] (No Exited Users as Exited ones) out of :red[$3979$]. The model might improve by relocating `(prediction_probability)` treshold.')

with col6:
    labels = ['0 (No Exited)', '1 (Exited)']
    fig_cm = px.imshow(
        matrix,
        x=labels,
        y=labels,
        text_auto=True,
        labels=dict(x="Prediction", y="Real Value", color="Frecuencia"),
        color_continuous_scale='purples'
    )
    fig_cm.update_xaxes(title_text='Prediction')
    fig_cm.update_yaxes(title_text='Real Value', autorange="reversed")
    fig_cm.update_layout(title_text='Confussion Matrix')
    st.plotly_chart(fig_cm)