import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('heart.csv')

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Different n_neighbors values to find best accuracy score
n_neighbors_list = [1, 3, 5, 7, 9, 12, 20, 25]  

for k in n_neighbors_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    accuracy = knn.score(X_test, y_test)
    print(f"Accuracy with n_neighbors={k}: {accuracy:.4f}")
