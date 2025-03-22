from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the last dimension
    x1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x2 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Apply depthwise separable convolutional layers with varying kernel sizes
    x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x1)
    x1 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x1)
    x1 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x1)
    x2 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x2)
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x2)
    x2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x2)
    x3 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(x3)
    x3 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(x3)
    x3 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x3)

    # Concatenate the outputs from all groups
    x = Concatenate()([x1, x2, x3])

    # Apply batch normalization and flatten the result
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Apply multiple branches for feature extraction
    x1 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)
    x2 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    x3 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(input_layer)
    x4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    x5 = Conv2D(filters=16, kernel_size=(1, 1), activation='relu')(input_layer)
    x6 = Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_layer)
    x7 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(input_layer)

    # Concatenate the outputs from all branches
    x = Concatenate()([x1, x2, x3, x4, x5, x6, x7])

    # Apply batch normalization and flatten the result
    x = BatchNormalization()(x)
    x = Flatten()(x)

    # Apply three fully connected layers to produce the final classification
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=input_layer, outputs=x)

    return model