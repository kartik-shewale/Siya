import gensim.downloader as api
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
import spacy
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from gensim.models import KeyedVectors
import requests
import seaborn as sn
import json


###################### Word2vec Opertaions ############################

# wv = api.load('word2vec-google-news-300')
# wv.save("Siya_FakeNewsProject\\word2vec-google-news-300.model")

from gensim.models import KeyedVectors
model_path = r"Siya_FakeNewsProject\word2vec-google-news-300.model"
wv = KeyedVectors.load(model_path, mmap='r')

nlp = spacy.load("en_core_web_lg")


###################### Word2vec Opertaions End ############################

###################### Formating Data Opertaions ############################

df = pd.read_csv(r'Siya_FakeNewsProject\Training.csv')
df['label_num'] = df['label'].map({'Fake': 0, 'Real': 1})
df = df.dropna()
print("Shape of the dataframe after removing NaN values:", df.shape)
print("Distribution of labels:")
print(df['label'].value_counts())
df.head(5)

###################### Formating Data Opertaions End ############################

###################### Saving Formated Data As Training2 ############################

df.to_csv(r'Siya_FakeNewsProject\Training2.csv', index=False)
df = pd.read_csv(r'Siya_FakeNewsProject\Training2.csv')
print('Training2 loaded')

###################### Saving Formated Data As Training2 End ############################

###################### Preprocess and vectorize text ############################

df['label'].value_counts()

count=0
def preprocess_and_vectorize(text):
    global count
    # Process the text: remove stop words and lemmatize
    doc = nlp(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Handle out-of-vocabulary (OOV) tokens
    
    valid_tokens = [token for token in filtered_tokens if token in wv]
    count += 1
    print(count)
    if valid_tokens:
        # Calculate the mean vector for the valid tokens
        return np.mean([wv[token] for token in valid_tokens], axis=0)
    else:
        # If no valid tokens, return a zero vector of the same size as the word vectors
        return np.zeros(wv.vector_size)

df['vector'] = df['Text'].apply(preprocess_and_vectorize)
print('Token Validated')

# Filter out rows with empty vectors (optional but recommended)
df = df[df['vector'].apply(lambda x: x is not None and np.any(x != 0))]

print("Shape of DataFrame after filtering out empty vectors:", df.shape)

###################### Preprocess and vectorize text End ############################

###################### Creating Test Data ############################
# Function to create X and y for training/testing
print('create_test_data')

def create_test_data(df, vector_column, label_column):
    # Stack vectors into a 2D array (features) and extract labels
    X_test_2d = np.stack(df[vector_column].values)
    y_test = df[label_column].values
    print("Shape of X_test after reshaping: ", X_test_2d.shape)
    print("Shape of y_test: ", y_test.shape)
    return X_test_2d, y_test

# Create test data
X_DF_2d, y_DF = create_test_data(df, 'vector', 'label_num')

###################### Creating Test Data End ############################

####################### Save vectors and labels into a pickle file ############################

vector_data = {'vector': X_DF_2d, 'label_num': y_DF}
print('Model Creating')

with open('Siya_FakeNewsProject\\vector_data.pkl', 'wb') as f:
    pickle.dump(vector_data, f)

print("Vector data saved as 'vector_data.pkl'")

####################### Save vectors and labels into a pickle file End ############################

####################### Loading Saved Vector Data ############################

with open('Siya_FakeNewsProject\\vector_data.pkl', 'rb') as f:
    vector_data = pickle.load(f)

# print("Loaded data shapes:")
# print("Shape of X_DF_2d:", X_DF_2d.shape)
# print("Length of y_DF:", len(y_DF))

# Create a DataFrame from the loaded vector data
df = pd.DataFrame(vector_data['vector'], columns=[f'feature_{i}' for i in range(vector_data['vector'].shape[1])])
df['label_num'] = vector_data['label_num']

print("DataFrame created from vector data:")
print(df.head())

print("Columns in DataFrame:", df.columns)

X_test = df.drop('label_num', axis=1).values
y_test = df.label_num
X_test_2d =  np.stack(X_test)

# Save the data to a pickle file
data = {'X_test_2d': X_test_2d, 'y_test': y_test}

with open('Siya_FakeNewsProject\\test_data.pkl', 'wb') as f:
    pickle.dump(data, f)

#Saving The vector
df.to_csv('Siya_FakeNewsProject\\vector.csv', index=False)

####################### Loading Saved Vector Data End ############################

####################### Spliting data into train test ############################

X_train, X_test, y_train, y_test = train_test_split(
    df.drop('label_num', axis=1).values,
    df.label_num,
    test_size=0.1, # 10% samples will go to test dataset
    random_state=2022,
    stratify=df.label_num
)

print("Shape of X_train before reshaping: ", X_train.shape)
print("Shape of X_test before reshaping: ", X_test.shape)

X_train_2d = np.stack(X_train)
X_test_2d =  np.stack(X_test)

data2 = {'X_train_2d': X_train_2d, 'X_test_2d': X_test_2d}

with open('Siya_FakeNewsProject\\data2.pkl', 'wb') as f:
    pickle.dump(data2, f)

print("Training and test data saved as 'data.pkl'")
print("Shape of X_train after reshaping: ", X_train_2d.shape)
print("Shape of X_test after reshaping: ", X_test_2d.shape)

####################### Spliting data into train test ############################

####################### Evaluate And Saving Model #############################

def evaluate_and_save_model(model, model_name):
    model.fit(X_train_2d, y_train)
    y_pred = model.predict(X_test_2d)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} classification report")
    print(classification_report(y_test, y_pred))
    print("aaa")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('Truth')
    plt.title(f"{model_name} Confusion Matrix")
    plt.show()

    # Save the model
    model_path = os.path.join('Siya_FakeNewsProject\\all_models', f'{model_name}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return acc

 # Save the model
    model_path = os.path.join('Siya_FakeNewsProject\\all_models', f'{model_name}_model.pkl')
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return acc

####################### Evaluate And Saving Model ############################

######################## Train and evaluate various models ############################

from catboost import CatBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC

models = {
    # "AdaBoostClassifier": AdaBoostClassifier(),
    # "LGBMClassifier": LGBMClassifier(),
    # "CatBoostClassifier": CatBoostClassifier(verbose=0),
    # "ExtraTreeClassifier": ExtraTreeClassifier(),
    # "VotingClassifier": VotingClassifier(estimators=[('lr', LogisticRegression(max_iter=1000)),('rf', RandomForestClassifier()),('svc', SVC())], voting='hard'),
    # "StackingClassifier": StackingClassifier(estimators=[('rf', RandomForestClassifier()),('svc', SVC())], final_estimator=LogisticRegression()),
    # "BaggingClassifier_SVC": BaggingClassifier(estimator=SVC(), n_estimators=10),
    # "MLPClassifier": MLPClassifier(max_iter=1000),
    # "ExtraTreesClassifier": ExtraTreesClassifier(),
    # "LinearDiscriminantAnalysis": LinearDiscriminantAnalysis(),
    # "QuadraticDiscriminantAnalysis": QuadraticDiscriminantAnalysis(),
    # "RidgeClassifier": RidgeClassifier(),
    # "PassiveAggressiveClassifier": PassiveAggressiveClassifier(),
    # "Perceptron": Perceptron(),
    # "BaggingClassifier": BaggingClassifier(),
    # "HistGradientBoostingClassifier": HistGradientBoostingClassifier(),
    # "SGDClassifier": SGDClassifier(),
    # "NearestCentroid": NearestCentroid(),
    # "GradientBoostingClassifier": GradientBoostingClassifier(),
    # "LogisticRegression": LogisticRegression(max_iter=1000),
    # "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    # "SupportVectorClassifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "K-NeighborsClassifier": KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
    "RandomForestClassifier": RandomForestClassifier(),

}

# Create directory if it does not exist
if not os.path.exists('Siya_FakeNewsProject\\all_models'):
    os.makedirs('Siya_FakeNewsProject\\all_models')

# Evaluate and save all models
accuracies = {}
for model_name, model in models.items():
    acc = evaluate_and_save_model(model, model_name)
    accuracies[model_name] = acc

with open('Siya_FakeNewsProject\\accuracies.json', 'w') as f:
    json.dump(accuracies, f)

for model_name, accuracy in accuracies.items():
    print(f"Accuracy of {model_name}: {accuracy:.4f}")

######################## Train and evaluate various models End ############################

######################## TTraining our New Neural Network Model ############################

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


with open('Siya_FakeNewsProject\\vector_data.pkl', 'rb') as f:
    vector_data = pickle.load(f)

# Extract X (features) and y (labels)
X = vector_data['vector']
y = vector_data['label_num']

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build the Neural Network with improvements
model = Sequential()
# Input layer
model.add(Input(shape=(X_train.shape[1],)))

# First hidden layer with BatchNormalization
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Second hidden layer
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Third hidden layer
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model with a tuned learning rate
optimizer = Adam(learning_rate=0.0001)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tensorboard = TensorBoard(log_dir='logs', histogram_freq=1)

# Train the model with EarlyStopping and TensorBoard
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, 
          callbacks=[early_stopping, tensorboard])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
import json

accuracy_data = {'Neural Model Accuracy': accuracy}

with open('Siya_FakeNewsProject\\accuracy.json', 'w') as f:
    json.dump(accuracy_data, f)

# Save the model in the recommended Keras format
model.save('Siya_FakeNewsProject\\news_prediction_model.keras')

# Predict on the test set
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# Calculate and print classification metrics
print(classification_report(y_test, y_pred, zero_division=1))  # or set to 0

cm = confusion_matrix(y_test, y_pred)
print(cm)

######################## TTraining our New Neural Network Model ############################
