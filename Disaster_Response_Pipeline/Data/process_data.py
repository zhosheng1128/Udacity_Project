import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #merge datasets
    df = pd.merge(messages,categories, on='id')
    
    return df
    #pass

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    df_cat = df['categories'].str.split(pat = ';', expand = True)
    # # select the first row of the categories dataframe
    row = df_cat.iloc[0,:]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2] )
    # Change column names
    df_cat.columns = category_colnames
    # Iterate through the category columns in df_cat to keep only the last             character of each string (the 1 or 0)
    for column in df_cat:
    # set each value to be the last character of the string
        for i in range (len(df_cat[column])):
            df_cat[column][i] = df_cat[column][i][-1:]
    # drop the original categories column from `df`
    df=df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, df_cat], axis=1)
    # drop duplicates
    df = df.drop_duplicates()
    
    return df
    #pass


def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('MyDisasterResponse', con = engine, index=False)
    pass  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()