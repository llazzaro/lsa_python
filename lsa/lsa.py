"""
LSA(Latent Semantic Analisys) is a technique originally developed for solving
the problems of synonymy and polysemy in information retrieval.
LSA is based on the vector-space model, exploiting singular value 
decomposition.
This provides functions for calculate LSA


LSA paper
http://lsi.argreenhouse.com/lsi/papers/JASIS90.pdf

Summarization (see pag19 for LSA)
http://files.nothingisreal.com/publications/Tristan_Miller/miller03b.pdf
"""
import re
import nltk
import math
import numpy
#from numpy import * #TODO fix this import
#from numpy.linalg import svd
#from numpy.linalg.decomp import diagsvd
#from numpy.linalg import norm
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import scipy.io
from scipy import linalg

class CoocurenceMatrix(numpy.matrix):
    """This matrix knows the frequency of keyword in each document
        rows = keywords
        columns = documents
                  doc_1  doc_2  doc_3 ... doc_m
        keyword_1 
        keyword_2
        ...
        keyword_n
    """
    __keyword_index = {}

    @classmethod
    def create(cls,documents):
        """
            @param documents list of documents with meaning

            @return coocurrence matrix
        """
        num_cols = len(documents)
        words = words_list(" ".join(documents))
        num_rows = len(words)
        #create the matrix
        res = cls(numpy.zeros(shape=(num_rows, num_cols)))

        #words must be unique!
        for index, word in enumerate(words):
            res[index] = keyword_frequency_list(documents, word)
            res.__keyword_index[index] = word
        
        return res

    def keyword_index(self,keyword):
        """
            @param keyword to know which index

            @return The index of the row for the keyword.
                    None if the keyword does not exist.

            @raise IndexError if keyword does not exists
        """
        return [key for key,value in self.__keyword_index.items() \
                                                    if value == keyword ][0]
    
    def keyword_row(self,keyword):
        """
            @param keyword to get the row

            @return The row of the corresponding keyword with frecuencies in
                    each document. None is returned if keyword does not exist

            @raise IndexError if keyword does not exists
        """
        index = self.keyword_index(keyword)
        return self[index]

    def keywords(self):
        """Returns the keywords associated in each row
            @return keyword list
        """
        return self.__keyword_index.values()

    def __str__(self):
        res = ""
        rows, cols = self.shape
        for row_index in range(0,rows):
            res += "%s\t\t\t" % self.__keyword_index[row_index]
            for col_index in range(0,cols):
                res += "\t%s\t" % (self[row_index,col_index])
            res += "\n"
        return res

    
    def load_matrix(self,filename):
        """Loads a matrix from disk
        """
        dictionary = scipy.io.loadmat(filename) #this loads a dict
        self = dictionary['m'] #@TODO fix this

    #scipy support file save, but seems this is more performant??
    def save_matrix(self, filename):
        """Dumps matrix to disk
        """
        scipy.io.savemat(filename, {'m': self}) #this saves a dict

def remove_punctuation(document):
    """Revomes punctuation in document
        
        @param document string with some meaning

        @return document string without punctuations
    """
    punctuation = re.compile(r'[-.,?!:;()|0-9]')
    return punctuation.sub(' ',document)

def tokenize_and_lemmatize(document):
    """Lemmatize is the process of grouping together the different
    inflected forms of a word. Ex, better has good as its lemma.

    @param document A discrete representation of meaning. string or
                    bufer is allowed.

    @return A list of words not neccesary unique and without lemmas
    """
    res = []
    wnl = WordNetLemmatizer()
    for word in nltk.word_tokenize(remove_punctuation(document)):
        if word.lower() not in stopwords.words('english'):
            res.append(wnl.lemmatize(word.lower()))

    return res

def words_list(keywords):
    """Creates a list using lemmatize to avoid semantic duplicates

    @param keywords 

    @return a list of unique words, tokenized and without lemmas
    """
    return list(set(tokenize_and_lemmatize(keywords)))
        
def keyword_frequency_list(documents, keyword):
    """Creates a row of keyword frequency for each document as columns

    @param documents collection of documents with some meaning
    @param keyword word for calculate frequency

    @return A list with frequencies of the keyword in each document
    """
    freq_list = [0] * len(documents)
    for index, document in enumerate(documents):
        #freqs_doc = 
        #nltk.FreqDist(tokenize_and_lemmatize(document))  # TODO: optimize this

        for word in tokenize_and_lemmatize(document):
            if word == keyword:
                freq_list[index] += 1#freqs_doc.freq(keyword)

    return freq_list


def matrix_reduce_sigma(matrix, dimensions=1):
    """This calculates the SVD of the matrix, reduces it and 
        creates a reduced matrix.

        @params matrix the matrix to reduce
        @params dimensions dimensions to reduce. 

        @return matrix The reduced matrix
    """
    uu, sigma, vt = linalg.svd(matrix)
    rows = sigma.shape[0]
    cols = sigma.shape[1]

    #delete n-k smallest singular values 
    #delete ie settings to zero
    smallerBound = min(rows, cols)
    for index in xrange(smallerBound - dimensions, rows):
        sigma[index] = 0 
    
    #since sigma is a unidimensional array
    #convert it to a matrix 
    sigma_matrix = linalg.diagsvd(sigma, len(uu), len(vt))
    uu_sigma = numpy.dot(uu, sigma_matrix)
    uu_sigma_vt = numpy.dot(uu_sigma, vt)

    return uu_sigma_vt

def scalar(vector1, vector2):
    length = min(vector1.shape[0], vector2.shape[0])
    res = 0
    for i in range(0, length):
        res = res + vector1[i]*vector2[i]
    return res

def cos_vector(vector1, vector2):
    """Calculates the Cosine metric to find semantically similar documents
        cosine is calculated with this forumla:  
         lets call vector1 A and vector2 B
         since A*B = ||A||*||B|| * cos(a)
         cos(a) = (A * B) / (||A|| * ||B||)

    @param vector1 one of the vector used to calcule the cosine
    @param vector2 the other vector needed to calculate the cosine

    @return float the cosine of the two vectors
    """
    return math.acos(float(scalar(vector1, vector2) / \
                (linalg.norm(vector1) * linalg.norm(vector2) ) ))

def search(document, documents):
    """this calculates LSA. ex. keyword: drama and you give 
        shakespeare dramas as documents, its expected to get high ratings

        @param document string document to compare against the documents
        @param documents list of documents to query against kewords

        @return list of ratings (this should be an object instead of list?)
    """
    print 'document'
    print document
    freq_matrix = CoocurenceMatrix.create(documents)  #TODO : fix this
    print 'reduced'
    reduced_matrix = matrix_reduce_sigma(freq_matrix, dimensions=1)
    print reduced_matrix
    print '-----------'
    query_vector = CoocurenceMatrix.create(document)
    print 'query_vector'
    print query_vector
    print '-------'
    print query_vector.shape

    for word_row in reduced_matrix:
        print word_row
        print word_row.shape
        break
    ratings = [cos_vector(query_vector, word_row) for word_row in reduced_matrix ]
    print ratings
