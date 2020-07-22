import matplotlib.pyplot as plt
from termcolor import colored
from keras.datasets import cifar100

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

(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')

show_shapes(x_train, y_train, x_test, y_test)
print('\n******************************\n')
show_sample_image(x_train, y_train)

"""
Dataset of 50,000 32x32 color training images, labeled over 100 categories, and 10,000 test images.

Returns 2 types data:

x_train and x_test
uint8 array of RGB image data with shape (num_samples, 32, 32, 3).
uint8 is an unsigned integer (0 to 255).
The "3" here refers to the 3 RGB channels.
y_train and y_test
uint8 array of category labels (integers in range 0-99) with shape (num_samples, 1)
"""