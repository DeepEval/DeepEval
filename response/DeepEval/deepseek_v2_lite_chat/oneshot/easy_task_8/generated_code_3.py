import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Activation, Concatenate, Dropout, Flatten, Dense

def dl_model():
    # Number of classes
    num_classes = 10

    # Input layer
    inputs = Input(shape=(28, 28, 1))

    # Depthwise separable convolutional layer
    dw_conv = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(inputs)
    dw_conv_output = Activation('relu')(dw_conv)

    # 1x1 convolutional layer for feature extraction
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(dw_conv_output)

    # Dropout layers for regularization
    dropout1 = Dropout(rate=0.5)(dw_conv_output)
    dropout2 = Dropout(rate=0.5)(conv1x1)

    # Concatenate the outputs of the parallel paths
    concatenated_output = Concatenate()([dropout1, dropout2])

    # Fully connected layers
    flatten = Flatten()(concatenated_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=num_classes, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=inputs, outputs=output)

    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])