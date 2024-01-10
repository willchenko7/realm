import csv

def csv2lol(filepath):
    #read csv file into a list of lists
    with open(filepath, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        data_list = [row for row in csvreader]
    return data_list

def lol2csv(filepath,data):
    #write a list of lists to a csv files
    with open(filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        for row in data:
            csvwriter.writerow(row)
    return

def append2csv(filepath,new_row):
    # appending list to a csv file
    with open(filepath, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(new_row)
    return

def txt2csv(text_file_path,csv_file_path,n):
    #read text from a file
    with open(os.path.join("data",text_file_path), "r", encoding="utf-8") as input_file:
        text = input_file.read()
    #Goal: given a string of text, split it into a list of lines and then write it to a csv file
    #step 0: remove all new line characters
    text = text.replace("\n"," ")
    #step 1: split the text separated by spaces
    lines = text.split(" ")
    #step 2: split every n words into a new line
    new_lines = []
    for i in range(0,len(lines),n):
        new_lines.append('"' + " ".join(lines[i:i+n])+ '"' )
    #step 3: write the new lines to a csv file, 
    #ecnlose the each line with double quotes to prevent commas from being interpreted as a new column
    with open(os.path.join("data",csv_file_path), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(new_lines))
    return