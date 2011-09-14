#!/usr/bin/python
import nltk
import numpy
from numpy import *
#from numpy.linalg import svd
#from numpy.linalg.decomp import diagsvd
#from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import scipy.io
from scipy import linalg
#https://github.com/josephwilk/semanticpy/blob/master/lsa/lsa.py

def tokenize_and_lemmatize(document):
    res = []
    wnl = WordNetLemmatizer()
    for word in nltk.word_tokenize(document):
        if word.lower() not in stopwords.words('english'):
            res.append(wnl.lemmatize(word.lower()))

    return res

def words_vector(keywords):
    #wnl = nltk.stem.WordNetLemmatizer()
    #stopset = set(nltk.corpus.stopwords.words('english'))
    #remove duplicates
    return list(set(tokenize_and_lemmatize(keywords)))
        
def keyword_row(documents,keyword):
    vector_row = [0] * len(documents)
    index = 0
    for document in documents:
        freqs_doc = nltk.FreqDist(tokenize_and_lemmatize(document))  # TODO: optimize this
        if keyword in tokenize_and_lemmatize(document):
            vector_row[index] += 1#freqs_doc.freq(keyword)
        index += 1

    return vector_row


def create_coocurrence_matrix(documents):
    num_cols = len(documents)
    words = tokenize_and_lemmatize(" ".join(documents))
    num_rows = len(words)
    matrix_res = numpy.zeros(shape=(num_rows,num_cols))
    row_index = 0
    for word in words:
        matrix_res[row_index] = keyword_row(documents,word)
        row_index += 1
        
    return matrix_res

def load_matrix(filename):
    dictionary = scipy.io.loadmat(filename) #this loads a dict
    return dictionary['m'] #@TODO fix this

def save_matrix(matrix,filename):
    scipy.io.savemat(filename,{'m':matrix}) #this saves a dict

def matrix_reduce_sigma(matrix,dimensions=1):
    uu,sigma,vt = linalg.svd(matrix)
    #reduce dimensions (param)
    rows = sigma.shape[0]
    for index in xrange(rows-dimensions,rows):
        sigma[index] = 0 
    
    reduced_matrix =  dot(dot(uu,linalg.diagsvd(sigma,len(matrix),len(vt))),vt)

    return reduced_matrix


def load_ycombinator_docs():
    docs = []
    for file in glob.glob('ycombinator_docs'):
        f = open(fn,'r')
        text = f.read()
        f.close()

        clean_text = nltk.clean_html(text)
        docs.append(clean_text)

    return docs

def cos_vector(vector1,vector2):
    return float(dot(vector1,vector2) / linalg.norm(vector1) * linalg.norm(vector2) )

def search(keywords,documents):
    pass

def search(keywords,documents):
    freq_matrix = create_coocurrence_matrix(documents)  #TODO : fix this
    print 'Coocurrence'
    print freq_matrix
    print '-----------'
    print 'reduced'
    reduced_matrix = matrix_reduce_sigma(freq_matrix,dimensions=1)
    print reduced_matrix
    print '-----------'
    for keyword in keywords:
        query_vector = keyword_row(documents,keyword)
        print query_vector
    ratings = [cos_vector(query_vector,word_row) for word_row in reduced_matrix ]
    print ratings


def test():
    #test
    #keywords =  'bitcoin startup hacks iphone android'
    keywords = ['cat','white','dog','good']
    documents = ["The cat in the hat disabled", "A cat is a fine pet ponies.",
            "Dogs and cats make good pets.","I haven't got a hat."]

    search(keywords,documents)
    matrix=[[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], 
            [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    #
    #ycombinator_documents = load_ycombinator_docs()
    #create_matrix(keywords_vector,ycombinator_documents)

def main():
    test()


if __name__ == '__main__':
    main()
