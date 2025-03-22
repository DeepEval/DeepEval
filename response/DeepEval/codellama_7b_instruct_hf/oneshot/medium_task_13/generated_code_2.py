import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    # Define the second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    # Define the third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')

    # Define the pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')

    # Define the concatenate layer
    concat = Concatenate()

    # Define the batch normalization layer
    batch_norm = BatchNormalization()

    # Define the flatten layer
    flatten = Flatten()

    # Define the first fully connected layer
    fc1 = Dense(units=128, activation='relu')

    # Define the second fully connected layer
    fc2 = Dense(units=64, activation='relu')

    # Define the output layer
    output = Dense(units=10, activation='softmax')

    # Create the input layer
    input_layer = Input(shape=input_shape)

    # Create the model
    model = keras.models.Model(inputs=input_layer, outputs=output)

    # Add the layers to the model
    model.add(conv1)
    model.add(conv2)
    model.add(conv3)
    model.add(pool)
    model.add(concat)
    model.add(batch_norm)
    model.add(flatten)
    model.add(fc1)
    model.add(fc2)
    model.add(output)

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model