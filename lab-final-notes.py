1) Python Programming (including OOP)

from abc import ABC, abstractmethod

# 1. Abstract Base Class and Abstract Method
# Abstract class cannot be instantiated directly. It is meant to be inherited.
class Animal(ABC):  # Inheriting ABC (Abstract Base Class)
    @abstractmethod
    def sound(self):
        pass  # Abstract method that must be implemented by any subclass

# 2. Class Inheritance
class Dog(Animal):
    def sound(self):  # Overriding the abstract method in Animal class
        return "Woof!"

class Cat(Animal):
    def sound(self):  # Overriding the abstract method in Animal class
        return "Meow!"

# 3. Multiple Inheritance
class Person:
    def __init__(self, name):
        self.name = name

    def introduce(self):
        return f"Hello, my name is {self.name}."

class Worker:
    def __init__(self, job):
        self.job = job

    def work(self):
        return f"I'm working as a {self.job}."

class Employee(Person, Worker):
    def __init__(self, name, job):
        Person.__init__(self, name)
        Worker.__init__(self, job)

    def get_details(self):
        return f"{self.introduce()} {self.work()}"

# 4. Encapsulation (Private Attributes and Methods)
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.__balance = balance  # Private attribute, not accessible directly

    def deposit(self, amount):
        if amount > 0:
            self.__balance += amount
            return f"Deposited {amount}. New balance is {self.__balance}."
        else:
            return "Amount must be positive."

    def withdraw(self, amount):
        if amount <= self.__balance:
            self.__balance -= amount
            return f"Withdrew {amount}. New balance is {self.__balance}."
        else:
            return "Insufficient funds."

    def get_balance(self):  # Getter method to access private balance
        return self.__balance

# Polymorphism: Different classes, same method name, different behaviors
animals = [Dog(), Cat()]
for animal in animals:
    print(animal.sound())  # Each animal makes a different sound

# Creating objects and using the methods
dog = Dog()
cat = Cat()
print(dog.sound())  # Output: Woof!
print(cat.sound())  # Output: Meow!

# Using Multiple Inheritance
employee = Employee("Alice", "Engineer")
print(employee.get_details())  # Output: Hello, my name is Alice. I'm working as a Engineer.

# Using Encapsulation (BankAccount class)
account = BankAccount("John", 1000)
print(account.deposit(500))  # Output: Deposited 500. New balance is 1500.
print(account.withdraw(200))  # Output: Withdrew 200. New balance is 1300.
print(account.get_balance())  # Output: 1300





2) Graphical Visualization and Data Preprocessing
Basic Visualizations
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Sample Data
data = {'Category': ['A', 'B', 'C', 'D'], 'Values': [23, 45, 56, 78]}
df = pd.DataFrame(data)

# Line Plot
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.title('Line Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Histogram
sns.histplot(df['Values'], kde=True)
plt.title('Histogram')
plt.show()

# Scatter Plot
sns.scatterplot(x=[1, 2, 3, 4], y=[1, 4, 9, 16])
plt.title('Scatter Plot')
plt.show()

# Pie Chart
plt.pie([30, 20, 25, 25], labels=['A', 'B', 'C', 'D'], autopct='%1.1f%%')
plt.title('Pie Chart')
plt.show()

# Box Plot
sns.boxplot(x=df['Values'])
plt.title('Box Plot')
plt.show()

# Heatmap
corr_matrix = np.random.rand(5, 5)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap')
plt.show()

# Pairplot
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species')
plt.show()



Data Preprocessing Example
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Example dataset
data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50], 'Label': [0, 1, 0, 1, 0]}
df = pd.DataFrame(data)

# Handle missing data (if any)
df.fillna(df.mean(), inplace=True)

# Train-Test Split
X = df[['Feature1', 'Feature2']]
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Data Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)





3) Machine Learning
KNN (K-Nearest Neighbors)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# KNN Model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



KMeans Clustering
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# KMeans Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Plotting the clusters
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X')
plt.title('KMeans Clustering')
plt.show()



PCA (Principal Component Analysis)
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X = iris.data

# PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot the PCA result
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.title('PCA - 2D Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()



Evaluation Metrics Example
from sklearn.metrics import confusion_matrix, f1_score

# Example true and predicted labels
y_true = [0, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1]

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# F1 Score
f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)





4) Natural Language Processing (NLP)
Text Preprocessing (Tokenization, Stemming, Lemmatization)
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample text
text = "The quick brown fox jumped over the lazy dog."

# Tokenization
tokens = word_tokenize(text)
print("Tokens:", tokens)

# Stop Words Removal
stop_words = set(stopwords.words('english'))
tokens_no_stopwords = [word for word in tokens if word.lower() not in stop_words]
print("Tokens without stop words:", tokens_no_stopwords)

# Punctuation Removal
tokens_no_punctuation = [word for word in tokens_no_stopwords if word not in string.punctuation]
print("Tokens without punctuation:", tokens_no_punctuation)

# Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens_no_punctuation]
print("Stemmed Words:", stemmed_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens_no_punctuation]
print("Lemmatized Words:", lemmatized_words)



Text Vectorization (CountVectorizer and TFIDF)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Sample documents
docs = [
    "I love programming in Python",
    "Python programming is fun",
    "Machine learning is amazing"
]

# Count Vectorizer
count_vectorizer = CountVectorizer()
count_vectorized = count_vectorizer.fit_transform(docs)
print("Count Vectorizer Output (bag-of-words):")
print(count_vectorized.toarray())

# TFIDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorized = tfidf_vectorizer.fit_transform(docs)
print("TFIDF Vectorizer Output:")
print(tfidf_vectorized.toarray())

