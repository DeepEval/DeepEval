import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, ZeroPadding2D, Conv2DNormActivation
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)

    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: <1x1 convolution, 3x3 convolution>
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch1)
    
    # Branch 2: <1x1 convolution, 1x7 convolution, 7x1 convolution, 3x3 convolution>
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(1, 7), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(7, 1), padding='valid', activation='relu')(branch2)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(branch2)
    
    # Branch 3: Max Pooling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    
    # Dropout layer for each branch
    branch1 = keras.layers.Dropout(0.5)(branch1)
    branch2 = keras.layers.Dropout(0.5)(branch2)
    branch3 = keras.layers.Dropout(0.5)(branch3)
    
    # Concatenate outputs from all branches
    concatenated = Concatenate()([branch1, branch2, branch3])
    
    # Process through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model