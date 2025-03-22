from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.applications import VGG16


def dl_model():
    # Load the VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(28, 28, 1))

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add a new input layer
    input_layer = Input(shape=(28, 28, 1))

    # Add the convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv2 = Conv2D(32, (3, 3), activation='relu')(conv1)
    maxpool = MaxPooling2D((2, 2))(conv2)

    # Add the second convolutional block
    conv3 = Conv2D(64, (3, 3), activation='relu')(maxpool)
    conv4 = Conv2D(64, (3, 3), activation='relu')(conv3)
    maxpool = MaxPooling2D((2, 2))(conv4)

    # Add the flatten layer
    flatten = Flatten()(maxpool)

    # Add the fully connected layers
    fc1 = Dense(128, activation='relu')(flatten)
    fc2 = Dense(64, activation='relu')(fc1)
    output_layer = Dense(10, activation='softmax')(fc2)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Return the constructed model
    return model