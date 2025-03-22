import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications.resnet import ResNet

def dl_model():

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split input image into three groups along the channel dimension
    splitted_input = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply different convolutional kernels to each group
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(splitted_input[0])
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(splitted_input[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(splitted_input[2])

    # Concatenate outputs from each group
    concatenated = Concatenate()([conv1, conv2, conv3])

    # Apply batch normalization and flatten the result
    batch_norm = BatchNormalization()(concatenated)
    flattened = Flatten()(batch_norm)

    # Pass the flattened result through two fully connected layers for classification
    fc1 = Dense(units=128, activation='relu')(flattened)
    fc2 = Dense(units=10, activation='softmax')(fc1)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fc2)

    return model