from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Flatten, Dense, Add


def dl_model():

    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the convolutional layer
    conv_layer = Conv2D(32, (3, 3), activation='relu')(input_shape)

    # Define the batch normalization layer
    batch_norm_layer = BatchNormalization()(conv_layer)

    # Define the global average pooling layer
    pooling_layer = GlobalAveragePooling2D()(batch_norm_layer)

    # Define the first fully connected layer
    fc1 = Dense(64, activation='relu')(pooling_layer)

    # Define the second fully connected layer
    fc2 = Dense(32, activation='relu')(fc1)

    # Define the output layer
    output_layer = Dense(10, activation='softmax')(fc2)

    # Define the model
    model = Model(inputs=input_shape, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model