import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Initialize the input layer
    input_layer = Input(shape=(32, 32, 3))

    # First stage of convolution and max pooling for downsampling
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    max_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)

    # Second stage of convolution and max pooling for downsampling
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling1)
    max_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)

    # Additional convolutional and dropout layers
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling2)
    dropout1 = Dropout(rate=0.2)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout1)
    dropout2 = Dropout(rate=0.2)(conv4)

    # Upsampling with skip connections
    up1 = Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout2)
    up2 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up1)
    up3 = Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(up2)

    # 1x1 convolutional layer for probability outputs
    probs = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='softmax')(up3)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=probs)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model