import keras
from keras.layers import Input, AveragePooling2D, Conv2D, Flatten, Dense, Dropout


def dl_model():
    # Define the input layer with shape (28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))

    # Average pooling layer with 5x5 window and 3x3 stride for feature dimensionality reduction
    pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3))(input_layer)

    # 1x1 convolutional layer to enhance depth and introduce nonlinearity
    conv = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1))(pool)

    # Flatten the feature maps
    flatten = Flatten()(conv)

    # Two fully connected layers with dropout regularization
    fc1 = Dense(units=128, activation='relu')(flatten)
    dropout = Dropout(rate=0.2)(fc1)
    fc2 = Dense(units=10, activation='softmax')(dropout)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=fc2)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model