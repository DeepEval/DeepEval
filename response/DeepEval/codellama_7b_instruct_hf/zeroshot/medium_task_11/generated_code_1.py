import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, concatenate
from keras.models import Model


def dl_model():

    # Define the input shape and number of filters for the convolutional layers
    input_shape = (32, 32, 3)
    n_filters = 16

    # Define the model architecture
    inputs = Input(shape=input_shape)
    x = Conv2D(n_filters, (3, 3), activation='relu')(inputs)

    # Path 1: Global average pooling followed by two fully connected layers
    x1 = GlobalAveragePooling2D()(x)
    x1 = Dense(64, activation='relu')(x1)
    x1 = Dense(10, activation='softmax')(x1)

    # Path 2: Global max pooling followed by two fully connected layers
    x2 = GlobalMaxPooling2D()(x)
    x2 = Dense(64, activation='relu')(x2)
    x2 = Dense(10, activation='softmax')(x2)

    # Compute channel attention weights
    channel_attention = concatenate([x1, x2], axis=-1)
    channel_attention = Flatten()(channel_attention)
    channel_attention = Dense(n_filters, activation='sigmoid')(channel_attention)
    channel_attention = Reshape((n_filters, 1, 1))(channel_attention)

    # Apply channel attention weights to the original features
    x = multiply([x, channel_attention])

    # Extract spatial features using separate average and max pooling operations
    x1 = AveragePooling2D()(x)
    x2 = MaxPooling2D()(x)

    # Concatenate spatial features along the channel dimension
    x = concatenate([x1, x2], axis=-1)

    # Flatten and feed into a fully connected layer to produce the final output
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs=inputs, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    # Return the constructed model
    return model