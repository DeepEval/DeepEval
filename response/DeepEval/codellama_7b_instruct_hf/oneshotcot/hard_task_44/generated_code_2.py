import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define first block
    # Split input along channel dimension into three groups
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=3))(input_layer)

    # Extract features from each group using convolutional layers with varying kernel sizes
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(split_layer[0])
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(split_layer[1])
    conv3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(split_layer[2])

    # Concatenate outputs from each group
    concat_layer = Concatenate()([conv1, conv2, conv3])

    # Apply dropout to reduce overfitting
    dropout_layer = Dropout(rate=0.2)(concat_layer)

    # Define second block
    # Define four branches for feature fusion
    branch1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(dropout_layer)
    branch2 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(dropout_layer)
    branch3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(dropout_layer)
    branch4 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')(dropout_layer)

    # Concatenate outputs from each branch
    concat_layer = Concatenate()([branch1, branch2, branch3, branch4])

    # Apply batch normalization
    batch_norm_layer = BatchNormalization()(concat_layer)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm_layer)

    # Define fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create and return the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model