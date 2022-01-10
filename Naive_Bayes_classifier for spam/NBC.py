import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def load_data():
    # Load data from csv file
    # and divide the dataset 80/20
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df_for_tests = df.head()

    idx = np.arange(df.shape[0])
    np.random.shuffle(idx)

    train_set_size = int(df.shape[0] * 0.8)

    train_set = df.loc[idx[:train_set_size]]
    test_set = df.loc[idx[train_set_size:]]

    return train_set, test_set, df_for_tests





def clean_data(message):
    '''Clear data by re lib'''
    return re.sub(r'\s+', ' ', re.sub('[^A-Za-z0-9]', ' ', message)).lower()


def prep_for_model(train_set, test_set):
    '''use clear for all message in dataset '''
    train_set_x = train_set['v2'][:]
    train_set_y = train_set['v1'][:]
    test_set_x = test_set['v2'][:]
    test_set_y = test_set['v1'][:]
    train_set_x = np.array([(clean_data(train_set_x[i])).split() for i in (train_set_x.index)], dtype='object')
    train_set_y = np.array([(clean_data(train_set_y[i])) for i in (train_set_y.index)], dtype='object')
    test_set_x = np.array([(clean_data(test_set_x[i])).split() for i in (test_set_x.index)], dtype='object')
    test_set_y = np.array([(clean_data(test_set_y[i])) for i in (test_set_y.index)], dtype='object')

    return train_set_x, train_set_y, test_set_x, test_set_y





def categories_words(x_train, y_train):
    ''' we break the data into categories ham, spam, and all message '''
    all_words_list = []
    ham_words_list = []
    spam_words_list = []

    for i in range(x_train.size):
        all_words_list += x_train[i]
        if y_train[i] == 'ham':
            ham_words_list += x_train[i]
        else:
            spam_words_list += x_train[i]

    all_words_list = np.array(all_words_list)
    ham_words_list = np.array(ham_words_list)
    spam_words_list = np.array(spam_words_list)

    return all_words_list, ham_words_list, spam_words_list


class Naive_Bayes(object):
    """
    Parameters:
    -----------
    alpha: int
        The smoothing coeficient.
    """

    def __init__(self, alpha):
        self.alpha = alpha

        self.train_set_x = None
        self.train_set_y = None

        self.all_words_list = []
        self.ham_words_list = []
        self.spam_words_list = []

        self.ham_words_dict = {}
        self.spam_words_dict = {}

        self.prior_ham_prob = None
        self.prior_spam_prob = None


    def fit(self, train_set_x, train_set_y):

        self.train_set_x = train_set_x
        self.train_set_y = train_set_y

        self.all_words_list, self.ham_words_list, self.spam_words_list = categories_words(train_set_x, train_set_y)

        self.prior_spam_prob = self.spam_words_list.size/self.all_words_list.size
        self.prior_ham_prob = 1-self.prior_spam_prob
        y_size = train_set_y.size
        self.all_words_list = np.unique(self.all_words_list)

        ham_size = self.ham_words_list.size
        spam_size = self.spam_words_list.size
        for i in self.all_words_list:
            self.spam_words_dict[i] = np.log((np.count_nonzero(self.spam_words_list == i) + self.alpha)/(spam_size + self.alpha*y_size))
            self.ham_words_dict[i] = np.log((np.count_nonzero(self.ham_words_list == i) + self.alpha)/(ham_size + self.alpha * y_size))

    def predict(self, test_set_x):

        prediction = []
        all_size = self.all_words_list.size
        ham_size = self.ham_words_list.size
        spam_size = self.spam_words_list.size

        for i in test_set_x:
            c = np.array([np.log(self.prior_spam_prob)])
            v = np.array([np.log(self.prior_ham_prob)])

            for j in i:

                if(j in (self.all_words_list)):
                    c = np.append(c, self.spam_words_dict[j])
                    v = np.append(v, self.ham_words_dict[j])

            prediction.append('ham' if (np.sum(v) > np.sum(c)) else 'spam')

        return np.array(prediction)


train_set, test_set, df_for_tests = load_data()

train_set_x, train_set_y, test_set_x, test_set_y = prep_for_model(train_set, test_set)

model = Naive_Bayes(alpha=1)
model.fit(train_set_x, train_set_y)
y_predictions = model.predict(test_set_x)
actual = list(test_set_y)
accuracy = (y_predictions == test_set_y).mean()
print(accuracy)
