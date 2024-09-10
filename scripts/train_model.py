import pickle
from sklearn.tree import DecisionTreeClassifier


with open('data/train_data.pkl', 'rb') as file:
    train_data = pickle.load(file)

with open('data/train_label.pkl', 'rb') as file:
    train_label = pickle.load(file)

model = DecisionTreeClassifier()

model.fit(train_data, train_label)
with open('models/model', 'wb') as file:
    pickle.dump(model, file)