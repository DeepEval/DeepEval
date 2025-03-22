from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 64))

    # 1x1 convolutional layer
    conv1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # 1x1 and 3x3 convolutional layers
    conv2_1 = Conv2D(32, (1, 1), activation='relu')(conv1)
    conv2_2 = Conv2D(64, (3, 3), activation='relu')(conv1)

    # Concatenate the feature maps
    x = Concatenate()([conv2_1, conv2_2])

    # Flatten the output feature map
    x = Flatten()(x)

    # Two fully connected layers
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model