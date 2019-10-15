import os
os.environ["PYTHONIOENCODING"] = "utf-8"
import webbrowser
import pandas as pd
import numpy as np
from numpy import *
from lxml import etree
from bs4 import BeautifulSoup
from flashtext.keyword import KeywordProcessor
from nltk import word_tokenize
import datetime,time
import json
import string
from collections import OrderedDict
from random import randint
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras import layers
from keras.layers import LSTM
from keras.utils import to_categorical


wordlist = list()
file = open("words.txt","r")
eng_words = set([line.rstrip("\n") for line in file])
#driver = webdriver.Chrome(r"C:\Users\YOGA\Downloads\chromedriver_win32\chromedriver.exe")

##count = 0
##html_count=0
##folderlist=list()

##

##for folderName,subfolders,filenames in os.walk(data_folder):
##    for folder in subfolders:
##        for file in os.listdir(data_folder+"/"+folder):
##            if file.endswith('html'):
##                #print(folder+"/"+file+"\n")
##                count+=1
##                print("Found",count,"HTML files\r",end = "")
##                file_list.append(folder+"/"+file)
                               
file_list = list()
infile = open(r"C:/Users/YOGA/Desktop/UIC/Sem 2/CS568/Project/unsub-pages/html_files","rb")
file_list = pickle.load(infile)

class_dict = OrderedDict()

class1 = ['do you want to','are you sure you want to','e-mail','email','mailing list','confirm','submit','about to unsubscribe','update preferences','save','manage subscription','end subscription','unsubscribing','unsubscribing from']
class2 = ['home','login','welcome','about','subscription']
class3 = ['success','successful','successfully','confirmed','no longer', 'receive','unsubscribed','have been unsubscribed','confirmation','preferences have been updated','saved','opted out','removed from','updated']
total_words = class1 + class2 + class3

master_processor = KeywordProcessor()
for word in total_words:
    master_processor.add_keyword(word)

class1_processor = KeywordProcessor()
for word in class1:
    class1_processor.add_keyword(word)

class2_processor = KeywordProcessor()
for word in class2:
    class2_processor.add_keyword(word)

class3_processor = KeywordProcessor()
for word in class3:
    class3_processor.add_keyword(word)

def percent(num1, num2):
    try:
        return 100 * float(num1) / float(num2)
    except:
        return 0

def find_class(text_from_html):
    total_extracted = set(master_processor.extract_keywords(text_from_html))
    class1_extracted = set(class1_processor.extract_keywords(text_from_html))
    class2_extracted = set(class2_processor.extract_keywords(text_from_html))
    class3_extracted = set(class3_processor.extract_keywords(text_from_html))
##    print((class1_extracted, class2_extracted, class3_extracted))
    percent1 = float(percent(len(class1_extracted),len(total_extracted)))
    percent2 = float(percent(len(class2_extracted),len(total_extracted)))
    percent3 = float(percent(len(class3_extracted),len(total_extracted)))
    
    if (total_extracted == 0) or (percent1 == percent2 == percent3 == 0):
        class_val = 'None'
    else:
        if percent1 >= percent2 and percent1 >= percent3:
            class_val = 1
        elif percent2 > percent1 and percent2 > percent3:
            class_val = 2
        elif percent3 >= percent1 and percent3 >= percent2:
            class_val = 3
    return class_val

wordset = list()
classes = list()
pages = list()

data_folder=r"C:/Users/YOGA/Desktop/UIC/Sem 2/CS568/Project/unsub-pages/output"
def page_to_text(file):
    tree = etree.parse(data_folder + "/" + file, etree.HTMLParser())
    html_code = etree.tostring(tree, method = "html")
    soup = BeautifulSoup(html_code, 'html.parser')
    texts = soup.body.findAll(text = True)
    text_from_html = " ".join(texts).lower().replace("\n"," ").replace("\t"," ").replace("\r"," ").translate(str.maketrans(string.punctuation.replace("-"," ")+string.digits," "*len(string.punctuation + string.digits)))
    return text_from_html
    
fcount = 0
for file in file_list[:1500]:
    fcount+=1
    print("Parsing file number",fcount,"\r",end = "")
    #print("\n",file)
    try:
        text_from_html = page_to_text(file)
        class_val = find_class(text_from_html)
        class_dict[file] = class_val
        if class_val != 'None':
            final_words = [word for word in word_tokenize(text_from_html) if word in eng_words]
            wordset.extend(final_words)
            pages.append((final_words, class_val))
            if class_val not in classes:
                classes.append(class_val)
        else:
            continue
    except:
        continue
    
print("\n")


wordset = list(set(wordset))
classes = list(set(classes))

print(len(pages),"documents")
print(len(classes),"classes")

##iterator = iter(class_dict.items())
##for i in range(20):
##    print(next(iterator))



training = list()
output = list()
output_empty = [0] * len(classes)
print("\n")
count = 0
for page in pages:
    count+=1
    print("Building BOW for file",count,"\r",end = "")
    try:
        bag_of_words = list()
        page_words = page[0]
        for word in wordset:
            bag_of_words.append(1) if word in page_words else bag_of_words.append(0)
        training.append(bag_of_words)
        output_row = list(output_empty)
        output_row[classes.index(page[1])] = 1
        output.append(output_row)
    except:
        continue

output_fin = list()
for i in output:
    a = i.index(1)+1
    output_fin.append(a)
    
X = np.array(training)
y = np.array(output_fin)
y_categorical = np.array(output)


x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
classifier = LogisticRegression(solver = 'lbfgs',multi_class = 'multinomial')
classifier.fit(x_train,y_train)
score = classifier.score(x_test,y_test)
print("Validation aacuracy for Scikit-Learn classifier is",score)

print("\n")
X_train, X_test, Y_train, Y_test = train_test_split(X,y_categorical, test_size = 0.2, random_state = 42)
model = Sequential()
model.add(layers.Dense(32, input_dim = X_train.shape[1], activation = 'relu'))
model.add(layers.Dense(32, activation = 'relu'))
model.add(layers.Dense(16, activation = 'relu'))
model.add(layers.Dense(8,activation = 'relu'))
model.add(layers.Dense(3,activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy',optimizer = 'adam', metrics = ['accuracy'])


model.fit(X_train, Y_train,epochs = 35,verbose = 1,validation_data = [X_test,Y_test],batch_size = 32)

training_loss,training_accuracy = model.evaluate(X_train,Y_train,verbose = False)
print("Training accuracy for Neural Network is",training_accuracy)
testing_loss,testing_accuracy = model.evaluate(X_test,Y_test,verbose = False)
print("Testing accuracy for Neural Network is",testing_accuracy)

#model.save("weights.hdf5")


testing = list()
testing_output = list()
test_op_instance = [0] * len(classes)

    
for page in file_list[int(len(file_list)/2):int(len(file_list)/2) + 600]:
    try:
        class_val = find_class(page_to_text(page))
        if class_val != 'None':
            bag_of_words = build_bag_of_words(page,wordset)
            testing.append(bag_of_words)
            output_row = list(test_op_instance)
            output_row[classes.index(class_val)] = 1
            testing_output.append(output_row)
        else:
            continue
    except:
        continue

testing_data_X = np.array(testing)
testing_data_Y = np.array(testing_output)

print(model.evaluate(testing_data_X, testing_data_Y))
    

def build_bag_of_words(page, wordset):
    # tokenize the page
    text_from_html = page_to_text(page)
    final_words = [word for word in word_tokenize(text_from_html) if word in eng_words]
    # bag of words
    bag = list()
    for word in wordset:
        bag.append(1) if word in page_words else bag.append(0)
                
    return bag
##
##def analyze(page):
##    x = build_bag_of_words(page, wordset)
##    
##    # input layer is our bag of words
##    input_layer = x
##    # matrix multiplication of input and hidden layer
##    hidden_layer_output = sigmoid(np.dot(input_layer, layer_1_weights))
##    # output layer
##    output_layer = sigmoid(np.dot(hidden_layer_output, layer_2_weights))
##
##    return output_layer
##

#print(count)
