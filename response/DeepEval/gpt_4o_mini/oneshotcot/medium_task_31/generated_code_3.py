import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # Step 2: Split the input image into three groups along the channel dimension
    split_layers = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Step 3: Define the convolutional layers for each split
    path1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layers[0])
    path2 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split_layers[1])
    path3 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split_layers[2])

    # Step 4: Concatenate the outputs of the convolutional layers
    merged = Concatenate()([path1, path2, path3])

    # Step 5: Flatten the concatenated features
    flatten_layer = Flatten()(merged)

    # Step 6: Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=10, activation='softmax')(dense1)  # 10 classes for CIFAR-10

    # Step 7: Build the model
    model = keras.Model(inputs=input_layer, outputs=dense2)

    return model