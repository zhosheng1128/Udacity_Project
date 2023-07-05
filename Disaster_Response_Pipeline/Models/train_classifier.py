# import libraries
import sys
import pickle
import pandas as pd
import numpy as np
import nltk 
from sqlalchemy import create_engine
from nltk.tokenize import RegexpTokenizer, wordpunct_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix 

''' This method is to load refined data to dataframe for running the model'''
def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('MyDisasterResponse', con = engine)
    
    # To find the output variable that has more than 2 categories
    L = []
    for col in df.iloc[:,4:].columns:
        if len(df[col]. unique()) > 2:
            L.append(col)
    
    # Remove any value other than 1 or 0 for that specific column
    for col in L:
        df = df[(df[col] == '1') | (df[col] == '0')]
        
    # Change object type to int tpye
    df.iloc[:,4:] = df.iloc[:,4:].astype(str).astype(int)
          
    X = df['message']
    Y = df.iloc[:,4:]
    return X, Y

def tokenize(text):
    ''' This method is to split each sentence to single word and remove some unimportant words as well'''
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    token = tokenizer.tokenize(text.lower())
    token = [item for item in token if item not in stopwords.words("english")]
    token = [WordNetLemmatizer().lemmatize(item, pos ='a') for item in token]
    return token

def build_model():
    ''' This method is to build pipeline for training model and create gridsearch for tuning model parameters'''
    # Build a pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf', RandomForestClassifier()) 
    ])
    
    # Create gridsearch to tune the model
    parameters = {
    'clf__n_estimators' : [50,100],
    'clf__criterion' : ['gini']
    }
    cv = GridSearchCV(pipeline, param_grid = parameters )
    
    return cv

def evaluate_model(model, X_test, Y_test):
    ''' This method is evaluate the given model by showing the average of accuracy, selected parameters, and classification reports'''
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    report = classification_report(Y_test, Y_pred)
    params = model.best_params_
    print('model accuracy is {}\n'.format(accuracy))
    print('the best parameters are {}\n'.format(params))
    print('the classificaiton report is {}'.format(report))
    
def save_model(model, model_filepath):
    file = open (model_filepath, 'wb')
    pickle.dump(model, file)
    file.close()
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()