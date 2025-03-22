import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, DepthwiseSeparableConv2D, BatchNormalization, Add
from tensorflow.keras.models import Model


def dl_model():
    
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the depthwise separable convolutional layer with 7x7 kernel size
    depthwise_separable_conv = DepthwiseSeparableConv2D(
        filters=32, kernel_size=(7, 7), strides=(1, 1), padding='same'
    )

    # Define the batch normalization layer
    batch_normalization = BatchNormalization()

    # Define the fully connected layers
    dense_1 = Dense(64, activation='relu')
    dense_2 = Dense(128, activation='relu')
    dense_3 = Dense(10, activation='softmax')

    # Define the model
    model = Model(
        inputs=Input(shape=input_shape),
        outputs=dense_3(Add()([dense_2(batch_normalization(depthwise_separable_conv(input_shape))), input_shape]))
    )

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    return model