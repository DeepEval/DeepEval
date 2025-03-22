import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.cifar10 import Cifar10

def dl_model():
    # Load the CIFAR-10 dataset
    cifar10 = Cifar10()
    input_layer = Input(shape=(32, 32, 3))

    # Split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Apply feature extraction on each channel group
    features_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    features_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    features_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])

    # Concatenate the output from each channel group
    concatenated_features = Concatenate()([features_1, features_2, features_3])

    # Apply batch normalization and flatten the result
    bath_norm = BatchNormalization()(concatenated_features)
    flatten_layer = Flatten()(bath_norm)

    # Pass the flattened result through three fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model