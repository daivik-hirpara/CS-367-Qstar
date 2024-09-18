import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data'
column_names = ['Class', 'T3', 'T4', 'TSH']
data = pd.read_csv(url, names=column_names, delim_whitespace=True)

class_map = {1: 'Hyperthyroid', 2: 'Hypothyroid', 3: 'Normal'}
data['Class'] = data['Class'].map(class_map)

data.fillna('Missing', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['Class']), data['Class'], test_size=0.3, random_state=42)
X_train['Class'] = y_train

model = BayesianNetwork([('T3', 'Class'), ('T4', 'Class'), ('TSH', 'Class')])
model.fit(X_train, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

def predict(model, X_test):
    y_pred = []
    for _, sample in X_test.iterrows():
        evidence = sample.to_dict()
        try:
            query = inference.map_query(variables=['Class'], evidence=evidence)
            y_pred.append(query['Class'])
        except KeyError as e:
            print(f"KeyError: {e} - Value may be missing in the CPD. Skipping sample.")
            y_pred.append('Unknown')
    return y_pred

y_pred = predict(inference, X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
