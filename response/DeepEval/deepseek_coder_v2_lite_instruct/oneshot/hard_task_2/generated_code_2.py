import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Add, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the convolutional paths for each group
    def conv_path(input_tensor):
        path1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(path2)
        return path3

    # Apply the convolutional paths to each group
    paths = [conv_path(group) for group in split]

    # Combine the outputs from the three paths using addition
    combined = Add()(paths)

    # Add the combined path back to the original input
    added = Add()([combined, input_layer])

    # Flatten the combined features
    flatten_layer = Flatten()(added)

    # Pass the flattened features through a fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model