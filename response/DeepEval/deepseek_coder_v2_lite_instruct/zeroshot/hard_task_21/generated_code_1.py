import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, Add, Lambda, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))

    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_path = SeparableConv2D(32, (3, 3), activation='relu')(main_path)
    main_path = SeparableConv2D(32, (5, 5), activation='relu')(main_path)
    main_path_outputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(main_path)

    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)

    # Concatenate outputs from main path
    concatenated_outputs = Add()([main_path_outputs[0], main_path_outputs[1], main_path_outputs[2], branch_path])

    # Flatten and add fully connected layers
    flattened = Flatten()(concatenated_outputs)
    outputs = Dense(10, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])