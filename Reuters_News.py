import matplotlib.pyplot as plt
from termcolor import colored
from keras.datasets import reuters

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

(x_train, y_train), (x_test, y_test) = reuters.load_data()

show_shapes(x_train, y_train, x_test, y_test)
print('\n******************************\n')
show_sample(x_train, y_train, idx=1)

"""
Dataset of 11,228 newswires from Reuters, labeled over 46 topics. 
As with the IMDB dataset, each wire is encoded as a sequence of word indexes (same conventions).

Returns 2 types data:

x_train and x_test
list of sequences, which are lists of indexes (integers).
y_train and y_test
list of integer labels (0 to 45).
"""