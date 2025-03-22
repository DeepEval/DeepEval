from keras.models import Model
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout



def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)

    # Define the average pooling layer with a 5x5 window and a 3x3 stride
    average_pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), padding='same')(input_shape)

    # Define the 1x1 convolutional layer
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(average_pooling)

    # Define the flatten layer
    flatten = Flatten()(conv1x1)

    # Define the fully connected layers with dropout
    fc1 = Dense(64, activation='relu')(flatten)
    fc2 = Dense(10, activation='softmax')(fc1)

    # Define the model
    model = Model(inputs=input_shape, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

    return model