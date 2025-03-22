import keras
from keras.layers import Input, GlobalAveragePooling2D, Dense, Flatten, concatenate
from keras.models import Model


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling
    gap = GlobalAveragePooling2D()(input_layer)
    fc1 = Dense(units=128, activation='relu')(gap)
    fc2 = Dense(units=128, activation='relu')(fc1)
    reshape = Flatten()(fc2)

    # Block 2: Convolutional Layers
    conv1 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2)(conv2)

    # Branch from Block 1
    branch = concatenate([reshape, max_pooling], axis=1)

    # Fusion
    fusion = concatenate([conv1, branch], axis=1)

    # Fully Connected Layers
    fc3 = Dense(units=128, activation='relu')(fusion)
    fc4 = Dense(units=10, activation='softmax')(fc3)

    # Define the model
    model = Model(inputs=input_layer, outputs=fc4)

    # Compile the model
    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    # Return the constructed model
    return model