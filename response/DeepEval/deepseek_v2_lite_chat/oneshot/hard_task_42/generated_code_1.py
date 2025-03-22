import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.models import Model
from keras.utils import to_categorical

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Define the input shape
input_layer = Input(shape=(28, 28, 1))

# Block 1
def block1():
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(conv3)
    return pool1

pool1_output = block1()

# Dropout and flattening
dropout1 = keras.layers.Dropout(0.5)(pool1_output)
flatten = Flatten()(dropout1)

# Block 2
def block2():
    path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten)
    path2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten)
    path3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten)
    path4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten)
    conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(flatten)
    conv1x1_pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1x1)

    concat = Concatenate(axis=-1)([path1, path2, path3, path4, conv1x1_pool])
    return concat

concat_output = block2()

# Batch normalization and dense layers
bn = BatchNormalization()(concat_output)
dense1 = Dense(units=128, activation='relu')(bn)
dense2 = Dense(units=64, activation='relu')(dense1)
output_layer = Dense(units=10, activation='softmax')(dense2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

return model