import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    # Split the input image into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    # Apply different convolutional kernels to each group
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(groups[1])
    conv3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(groups[2])
    # Concatenate the outputs from the three groups
    concatenated = Concatenate()([conv1, conv2, conv3])
    # Flatten the concatenated output
    flattened = Flatten()(concatenated)
    # Pass the flattened output through two fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=10, activation='softmax')(dense1)
    model = keras.Model(inputs=input_layer, outputs=dense2)
    return model