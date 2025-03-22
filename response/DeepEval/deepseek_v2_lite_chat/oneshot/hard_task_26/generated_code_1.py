import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)

    # Branch 1 - 3x3 convolutional layer
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool)
    
    # Branch 2 - Max pooling, 3x3 convolutional layer, and upsampling
    branch2 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3 - Max pooling, 5x5 convolutional layer, and upsampling
    branch3 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(64, (5, 5), activation='relu', padding='same')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate branch outputs
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Additional 1x1 convolutional layer to match the output channels
    conv2 = Conv2D(64, (1, 1), activation='relu')(concat)

    # Final 1x1 convolutional layer
    conv = Conv2D(10, (1, 1), activation='softmax')(conv2)

    model = Model(inputs=input_layer, outputs=conv)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model