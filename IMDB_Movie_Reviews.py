import matplotlib.pyplot as plt
from termcolor import colored
from keras.datasets import imdb

def show_shapes(x_train, y_train, x_test, y_test, color='green'):
    print(colored('Training shape:', color, attrs=['bold']))
    print('  x_train.shape:', x_train.shape)
    print('  y_train.shape:', y_train.shape)
    print(colored('\nTesting shape:', color, attrs=['bold']))
    print('  x_test.shape:', x_test.shape)
    print('  y_test.shape:', y_test.shape)
def plot_data(my_data, cmap=None):
    plt.axis('off')
    fig = plt.imshow(my_data, cmap=cmap)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    print(fig)
def show_sample(x_train, y_train, idx=0, color='blue'):
    print(colored('x_train sample:', color, attrs=['bold']))
    print(x_train[idx])
    print(colored('\ny_train sample:', color, attrs=['bold']))
    print(y_train[idx])
def show_sample_image(x_train, y_train, idx=0, color='blue', cmap=None):
    print(colored('Label:', color, attrs=['bold']), y_train[idx])
    print(colored('Shape:', color, attrs=['bold']), x_train[idx].shape)
    print()
    plot_data(x_train[idx], cmap=cmap)

(x_train, y_train), (x_test, y_test) = imdb.load_data()

show_shapes(x_train, y_train, x_test, y_test)
print('\n******************************\n')
show_sample(x_train, y_train, idx=123)

"""
Dataset of 25,000 movie reviews from IMDB, labeled by sentiment (positive/negative). 
Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data. 
This allows for quick filtering operations such as: "only consider the top 10,000 most common words, but eliminate the top 20 most common words".

As a convention, "0" does not stand for a specific word, but instead is used to encode any unknown word.

Returns 2 types data:

x_train and x_test
list of sequences, which are lists of indexes (integers).
y_train and y_test
list of integer labels (1 or 0).
"""