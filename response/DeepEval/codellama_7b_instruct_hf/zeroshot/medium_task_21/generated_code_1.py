from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(64, (1, 1), activation='relu')(input_layer)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(64, (3, 3), activation='relu')(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)
    branch3 = Conv2D(64, (3, 3), activation='relu')(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = MaxPooling2D((2, 2))(input_layer)
    branch4 = Conv2D(64, (1, 1), activation='relu')(branch4)

    # Concatenate the outputs from all branches
    x = Concatenate()([branch1, branch2, branch3, branch4])

    # Flatten the concatenated output
    x = Flatten()(x)

    # Add dropout layers to mitigate overfitting
    x = Dropout(0.25)(x)
    x = Dropout(0.25)(x)

    # Add two fully connected layers for classification
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Create and return the model
    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model