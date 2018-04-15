import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # implement the recognizer
    for index in range(test_set.num_items):

        # For test_word get (X, lengths) => (array of feature lists, length of sequence within X)
        test_word_X, test_word_lengths = test_set.get_item_Xlengths(index)
        # initialize
        best_likelihood = float("-inf")
        best_match =  None
        word_logLs = {}
        
        # Calculate LogL for test_word with model from models
        # Assign that logL value as probability of match with corresponding word from models
        for word, model in models.items():
            try:
                word_logLs[word] = model.score(test_word_X, test_word_lengths)
            except:
                # Assign -inf when model score could not be evaluated
                word_logLs[word] = float("-inf")
            # Update best/highest value of logLout of words processed from models and assing that word as best mathing word
            if word_logLs[word] > best_likelihood:
                best_likelihood = word_logLs[word]
                best_match  = word
                
        #populate return values
        probabilities.append(word_logLs)
        guesses.append(best_match)

    # return probabilities, guesses    
    return probabilities, guesses
    
    
