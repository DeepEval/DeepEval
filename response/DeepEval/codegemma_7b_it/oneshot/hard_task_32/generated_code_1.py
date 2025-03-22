import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():

    input_layer = Input(shape=(28, 28, 1))

    # Branch 1: Depthwise Separable Convolutional Layer followed by a 1x1 Convolutional Layer
    def block1(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
        return conv2

    # Branch 2: Depthwise Separable Convolutional Layer followed by a 1x1 Convolutional Layer with Dropout
    def block2(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', dropout=0.2)(conv1)
        return conv2

    # Branch 3: Depthwise Separable Convolutional Layer followed by a 1x1 Convolutional Layer with Dropout and Max Pooling
    def block3(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_initializer='he_normal')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', dropout=0.2)(conv1)
        maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv2)
        return maxpool

    # Concatenate outputs from all three branches
    branch1_output = block1(input_tensor)
    branch2_output = block2(input_tensor)
    branch3_output = block3(input_tensor)
    concat_output = Concatenate()([branch1_output, branch2_output, branch3_output])

    # Fully connected layers
    dense1 = Dense(units=64, activation='relu')(concat_output)
    dense2 = Dense(units=32, activation='relu')(dense1)

    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Model definition
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model