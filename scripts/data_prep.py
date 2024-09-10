from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

dataset = load_iris()
X = dataset.data
y = dataset.target

test_data,train_data,test_label,train_label = train_test_split(X,y,test_size=0.8)


with open('data/test_data.pkl', 'wb') as file:
    pickle.dump(test_data, file)
with open('data/train_data.pkl', 'wb') as file:
    pickle.dump(train_data, file)
with open('data/test_label.pkl', 'wb') as file:
    pickle.dump(test_label, file)
with open('data/train_label.pkl', 'wb') as file:
    pickle.dump(train_label, file)

print("succeed")