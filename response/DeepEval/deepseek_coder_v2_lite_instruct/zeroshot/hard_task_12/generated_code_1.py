import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Add, Flatten, Dense

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 64))

    # Main path
    main_path = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(input_layer)
    conv1x1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same')(main_path)
    conv3x3 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(main_path)
    main_output = tf.concat([conv1x1, conv3x3], axis=-1)

    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=3, strides=1, padding='same')(input_layer)

    # Combine main and branch paths
    combined_output = Add()([main_output, branch_path])

    # Flatten the output
    flattened_output = Flatten()(combined_output)

    # Add fully connected layers
    fc1 = Dense(units=128, activation='relu')(flattened_output)
    fc2 = Dense(units=10, activation='softmax')(fc1)  # Assuming 10 classes for classification

    # Define the model
    model = Model(inputs=input_layer, outputs=fc2)

    return model

# Example usage
model = dl_model()
model.summary()