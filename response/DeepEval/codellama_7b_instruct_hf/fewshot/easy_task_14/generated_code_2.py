import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Reshape, Flatten

 和 return model
def dl_model():
    # Define the input layer with a shape of (32, 32, 3)
    input_layer = Input(shape=(32, 32, 3))

    # Compress the input features with global average pooling
    gap = GlobalAveragePooling2D()(input_layer)

    # Define two fully connected layers with the same number of output channels as the input
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=128, activation='relu')(fc1)

    # Reshape the weights to align with the input shape
    reshape_weights = Reshape((3, 3, 128))(fc2)

    # Multiply the weights with the input feature map
    elementwise_mul = keras.layers.Multiply()([input_layer, reshape_weights])

    # Flatten the result and pass it through another fully connected layer
    flattened = Flatten()(elementwise_mul)
    dense = Dense(units=10, activation='softmax')(flattened)

    # Define the model
    model = keras.models.Model(inputs=input_layer, outputs=dense)

    # Compile the model with a loss function and an optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model