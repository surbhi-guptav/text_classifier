import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

# Loading the data
data = pd.read_csv("E:\F drive data\Downloads\Exp-Learning\EPICS\Dataset\CSV\Final_dataset_Shuffled_data.csv")

# Splitting the data
X_train, X_test, Y_train, Y_test = train_test_split(data['Text'], data['Label'], test_size=0.2, random_state=42)

# Vectorizing the text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Printing shapes to verify
print(f"X_train shape: {X_train_vec.shape}")
print(f"X_test shape: {X_test_vec.shape}")
print(f"Y_train length: {len(Y_train)}")
print(f"Y_test length: {len(Y_test)}")

# Training the model
Model = MultinomialNB()
Model.fit(X_train_vec, Y_train)

# Saving the model and vectorizer
joblib.dump(Model, "text_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
