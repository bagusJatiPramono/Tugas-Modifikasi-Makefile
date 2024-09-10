import pickle
from sklearn.metrics import accuracy_score, classification_report

with open(r'models\model', 'rb') as file:
    model = pickle.load(file)

with open('data/test_data.pkl', 'rb') as file:
    test_data = pickle.load(file)
with open('data/test_label.pkl', 'rb') as file:
    test_label = pickle.load(file)

y_pred = model.predict(test_data)

accuracy = accuracy_score(test_label,y_pred)
print(classification_report(test_label,y_pred))
print("Accuracy:", accuracy)