'''
goal: read file from data/ , add a label column as the first column (default value of 0), and save to labeled_data/
'''
import os
from csv_helpers import lol2csv, csv2lol

def prepare_for_label(file_name):
    #read file
    lol = csv2lol(os.path.join('generated_data',file_name))
    #add label column
    for row in lol:
        row.insert(0,0)
    #save to labeled_data/
    lol2csv(os.path.join('labeled_data',file_name),lol)
    return 

if __name__ == '__main__':
    file_name = 'model5.csv'
    prepare_for_label(file_name)