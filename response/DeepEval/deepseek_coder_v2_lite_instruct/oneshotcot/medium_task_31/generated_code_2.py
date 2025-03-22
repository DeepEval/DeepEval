import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    # Define the three groups with different convolutional kernels
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Concatenate the outputs of the three groups
    concatenated_output = Concatenate()([group1, group2, group3])

    # Batch normalization and flattening
    batch_norm = BatchNormalization()(concatenated_output)
    flatten_layer = Flatten()(batch_norm)

    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])