import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Define the input layer with a shape of (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Average pooling layer with a 5x5 window and a 3x3 stride
    pooling = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # Convolutional layer with a 1x1 kernel and a sigmoid activation function
    conv = Conv2D(filters=64, kernel_size=(1, 1), activation='sigmoid')(pooling)

    # Flatten the feature maps
    flatten = Flatten()(conv)

    # Two fully connected layers with dropout regularization
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=10, activation='softmax')(dense1)

    # Define the output layer with a shape of (10,) for 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model by connecting the layers
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model with a loss function, optimizer, and metrics
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model