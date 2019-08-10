import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
df = pd.read_csv("passengers.csv")

# Update sex column to numerical
#print(df.head())
# titanic['Sex'] = titanic['Sex'].apply(lambda x: 1 if x == 'female' else 0)
df['Sex'] = df['Sex'].map({'female':1, 'male':0})
#print(df.head())


# Fill the nan values in the age column
#print(df['Age'].values)

df['Age'].fillna(value=df.loc[:, "Age"].mean(), inplace=True)

#print(df['Age'].values)

# Create a first class column
df['FirstClass'] = df['Pclass'].apply(lambda x: 1 if x == 1 else 0)

# Create a second class column
df['SecondClass'] = df['Pclass'].apply(lambda x: 1 if x == 2 else 0)

# Select the desired features
features = df[['Sex', 'Age', 'FirstClass', 'SecondClass']]
survival = df['Survived']

# Perform train, test, split
X_train, X_test, y_train, y_test = train_test_split(features, survival, test_size=0.8)

# Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Score the model on the train data
score_train = classifier.score(X_train, y_train)
print(score_train)

# Score the model on the test data
score_test = classifier.score(X_test, y_test)
print(score_test)

# Analyze the coefficients
print(classifier.coef_)

# Sample passenger features
Jack = np.array([0.0, 20.0, 0.0, 0.0])
Rose = np.array([1.0, 17.0, 1.0, 0.0])
Me = np.array([0.0, 22.0, 0.0, 1.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, Me])

# Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print(sample_passengers)

# Make survival predictions!
verdict = classifier.predict(sample_passengers)
verdict_prob = classifier.predict_proba(sample_passengers)

print(verdict)
print(verdict_prob)
