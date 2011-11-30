"""
 Unit tests for lsa functions   
"""
import unittest
import nltk

from lsa import tokenize_and_lemmatize
from lsa import words_list
from lsa import CoocurenceMatrix
from lsa import search

class TextCoocurrenceMatrix(unittest.TestCase):

    def test_instantiation(self):
        document_a = 'The cat (Felis catus), also known as the domestic cat '\
                     'or housecat to distinguish it from other felids and'\
                     ' felines, is a small, usually furry, domesticated, '\
                     'carnivorous mammal that is valued by humans for its '\
                     'companionship and for its ability to hunt vermin and'\
                     ' household pests. '
        document_b = 'Since strong bonds of affection can be established'\
                    'between any pet and its owner, and given the length of'\
                    'time the two may be together, the process of choosing '\
                    'a cat can be difficult. Like any long term relationship'\
                    ', a certain degree of care and thought should be'\
                    'exercised in choosing an animal, since the relationship'\
                    ' is often of such emotional significance to both animal '\
                    'and human.some cats...cat'
        documents = [document_a, document_b]
        matrix = CoocurenceMatrix.create(documents)
        #no punctuations
        self.assertTrue(',' not in matrix.keywords())
        self.assertTrue('(' not in matrix.keywords())
        self.assertTrue(')' not in matrix.keywords())
        self.assertTrue(',' not in matrix.keywords())
        self.assertEquals(len(matrix.keywords()),53)
        self.assertEquals(matrix.keyword_row('cat')[0,0],2.0)

class TestLSA(unittest.TestCase):
    """Test for LSA functions
    """

    def test_tokenize_and_lemmatize(self):
        """
        """
        document = "foot feet"
        expected = ['foot', 'foot']
        res = tokenize_and_lemmatize(document)
        self.assertEquals(expected,res)

    def test_words_list(self):
        """Ensures words are lemmatized and are unique
        """
        document = "foot feet foot cars dogs car dog women"
        res = words_list(document)
        self.assertEquals(len(res), 4)
        self.assertTrue('foot' in res)
        self.assertTrue('car' in res)
        self.assertTrue('dog' in res)
        self.assertTrue('woman' in res)

    def test_coocurrence(self):
        pass

    def test_reduced_small(self):
        pass

    def test_search(self):
        documents = \
        ' '.join(list(nltk.corpus.gutenberg.words('shakespeare-caesar.txt')))[0:2000].split('.')
#        document_b = ' '.join(list(nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')))
#        document_c = ' '.join(list(nltk.corpus.gutenberg.words('shakespeare-macbeth.txt')))

        search([documents[-1]], documents)
        self.assertTrue(False)

    #test borrwed from: 
    #https://github.com/josephwilk/semanticpy/blob/master/lsa/lsa.py
    def test_reduced(self):
        matrix = [[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], 
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

    def test_lsa(self):
        #test
        #keywords =  'bitcoin startup hacks iphone android'
        keywords = ['cat','white','dog','good']
        documents = ["The cat in the hat disabled", "A cat is a fine pet ponies.",
            "Dogs and cats make good pets.","I haven't got a hat."]

        matrix=[[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 
                [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], 
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0], 
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
