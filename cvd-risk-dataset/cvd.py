import random
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import LearningRateScheduler, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

random_state = 6162

# Set the random seed for reproducible results
tf.random.set_seed(random_state)
random.seed(random_state)
np.random.seed(random_state)

# Load the data and create training and testing sets
def prep_data():
    # Load the data
    data = pd.read_csv('CVD_cleaned.csv')

    # Define the features and the labels
    target = 'Heart_Disease'
    features = data.drop(columns=[target])
    labels = data[target]

    # Specify categorical columns for one-hot encoding
    categorical_columns = [
        'Checkup', 'Exercise', 'General_Health', 'Skin_Cancer', 
        'Other_Cancer', 'Depression', 'Diabetes', 'Arthritis', 
        'Sex', 'Age_Category', 'Smoking_History'
    ]

    # Define numeric columns for standardization
    numeric_columns = [
        'Height_(cm)', 'Weight_(kg)', 'BMI', 
        'Alcohol_Consumption', 'Fruit_Consumption', 
        'Green_Vegetables_Consumption', 'FriedPotato_Consumption'
    ]

    # Create the preprocessing pipelines for both numeric and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),
            ('cat', OneHotEncoder(), categorical_columns)
        ])

    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

    features_transformed = pipeline.fit_transform(features)

    # Encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)

    # Apply SMOTE
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(features_transformed, encoded_Y)

    # Convert integers to one-hot encoding
    y_onehot = np_utils.to_categorical(y_resampled)

    # Split the data into training and testing sets    

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_onehot, test_size=0.2, random_state=random_state)

    # Return dense arrays
    return X_train, X_test, y_train, y_test

# Define the Keras model
def create_model(input_number):
    model = Sequential()
    model.add(Dense(32, input_dim=input_number, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    return model

lr_decay = 0.993
# Define Learning Rate Scheduler
def scheduler(epoch, lr):
    if epoch < 5:
        return 0.01
    else:
        return lr * lr_decay
    


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = prep_data()
    print(X_train, y_train)
    # Calculate the class weights
    y_train_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_integers), y=y_train_integers)
    class_weights = dict(enumerate(class_weights))

    model = create_model(X_train.shape[1])
    early_stop = EarlyStopping(monitor='val_loss', patience=300, min_delta=0.0001)
    lr_scheduler = LearningRateScheduler(scheduler)
    model.fit(X_train, y_train, epochs=1000, batch_size=4096, validation_data=[X_test, y_test] , verbose=1, callbacks=[early_stop, lr_scheduler], class_weight=class_weights)
    model.evaluate(X_test, y_test, verbose=1)
    y_pred_prob = model.predict(X_test)
    print(y_pred_prob)
    y_pred = (y_pred_prob[:, 1] >= 0.7).astype(int)
    y_true = np.argmax(y_test, axis=1)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    #roc_auc = roc_auc_score(y_true, y_pred_prob)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    #print("ROC AUC Score:", roc_auc)
    print("Accuracy:", accuracy)
    model.save('cvd_model.h5')