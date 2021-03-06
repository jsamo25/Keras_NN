import matplotlib.pyplot as plt
from termcolor import colored
from keras.datasets import boston_housing

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

(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
show_shapes(x_train, y_train, x_test, y_test)
print('\n******************************\n')
show_sample(x_train, y_train)

"""
This dataset contains 13 attributes of houses at different locations around the Boston suburbs in the late 1970s.
Targets are the median values of the houses at a location (in k$).
Note: load_data() returns two tuples of Numpy arrays. The first tuple represents the training x-y pairs while the second tuple represents the testing x-y pairs.
"""