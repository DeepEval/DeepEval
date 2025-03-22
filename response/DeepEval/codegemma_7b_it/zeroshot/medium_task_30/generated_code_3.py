from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import Input, Model

def dl_model():

    # Create the input layer
    inputs = Input(shape=(32, 32, 3))

    # Convolution and max pooling layers
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

    # Flatten the output of the pooling layers
    x = Flatten()(x)

    # Concatenate the flattened outputs
    concat_outputs = [x]

    # Fully connected layers
    for _ in range(2):
        concat_outputs.append(Dense(256, activation='relu')(concat_outputs[-1]))

    # Output layer
    outputs = Dense(10, activation='softmax')(concat_outputs[-1])

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model