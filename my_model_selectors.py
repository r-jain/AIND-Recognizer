import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("inf")
        best_model = None
        # The current word data - self.X, has numbr of data rows as the first dimension and the the second dimension /column is features.
        # number of features
        num_features = len(self.X[0])
        # iterate from the min number of components to max number of components+1
        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(num_states)
                # Log of likelihood
                logL = hmm_model.score(self.X, self.lengths)
                # Log of number of data elements
                logN = np.log(len(self.X))
                # Number of parameters
                parameters = num_states **2 + (2 * num_features * num_states) - 1
                # BIC score for current num_states
                bic_score = (-2 * logL) + (parameters * logN)
                # lower value of bic_score is better
                if bic_score < best_score:
                    # Update the best/least score
                    best_score =  bic_score
                    # Assign the hmm_model with num_states number of states as best model
                    best_model =  hmm_model
            except:
                pass
        # return the model with least/best bic_score
        return best_model if best_model is not None else self.base_model(self.n_constant)

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection based on DIC scores
        # Initialize
        best_score = float("-inf")
        best_model = None

        other_words = []
        for other_word in self.words:
            if other_word != self.this_word:
                other_words.append(other_word)

        # iterate from the min number of components to max number of components+1
        for num_states in range(self.min_n_components, self.max_n_components+1):
            try:
                hmm_model = self.base_model(num_states)
                # Log of likelihood of the current word
                logL = hmm_model.score(self.X, self.lengths)   
                # logL values of all the words other than current word      
                other_logL = []   
                for other_word in other_words:
                    # self.hwords[other_word][0] = other.X, self.hwords[other_word][1] = other.lengths for all the word other than the current word
                    # append logL values for other_words
                    other_logL.append(hmm_model.score(self.hwords[other_word][0], self.hwords[other_word][1]))
                # DIC score for current num_states
                dic_score = logL - np.mean(other_logL)
                # Higher value of dic_score is better
                if dic_score > best_score:
                    # Update the best/highest score
                    best_score =  dic_score
                    # Assign the hmm_model with num_states number of states as best model
                    best_model =  hmm_model
            except:
                pass
        # return the model with highest/best dic_score
        return best_model if best_model is not None else self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # implement model selection using CV
        # Initialize
        best_score = float("-inf")
        best_num_States = None

        # iterate from the min number of components to max number of components+1
        for num_states in range(self.min_n_components, self.max_n_components+1):
            # kist of logL scores for different folds sets generated for this num_states
            logL_scores = []
            try:
                # Split into KFolds only when there is sufficient data
                if len(self.sequences) > 2:
                    split_method = KFold(n_splits = 3, shuffle = False, random_state = None)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # generate train, test folds
                        train_X, train_lengths = combine_sequences(cv_train_idx, self.sequences)
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)
                        # fit model against train folds
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        # Score aginst test folds
                        logL_scores.append(hmm_model.score(test_X, test_lengths))
                else:
                    # fit model against all the data
                    hmm_model = self.base_model(num_states)
                    logL_scores.append(hmm_model.score(self.X, self.lengths))

                # mean of LogL for this num_states
                mean_logL = np.mean(logL_scores)
            except:
                pass

            # update best_score and best_num_states 
            if mean_logL > best_score:
                best_score = mean_logL
                best_num_States  = num_states

        # return the model for best_num_states (with best/highest mean_logL) fitted over all the data points
        return self.base_model(best_num_States) if best_num_States is not None else self.base_model(self.n_constant)