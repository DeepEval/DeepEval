import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the lambda layer to split the input image into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, 3, axis=3))(input_shape)

    # Define the separable convolutional layers for each channel group
    channel_groups = [
        Conv2D(64, (1, 1), activation='relu', input_shape=input_shape),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(256, (5, 5), activation='relu')
    ]

    # Define the fully connected layers
    x = Flatten()(channel_groups)
    x = Dense(512, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define the model
    model = keras.Model(inputs=input_shape, outputs=x)

    return model