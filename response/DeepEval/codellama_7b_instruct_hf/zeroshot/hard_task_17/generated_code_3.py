from keras.models import Model
from keras.layers import Input, GlobalAveragePooling2D, Flatten, Dense, Conv2D, MaxPooling2D, Add
from keras.applications import VGG16


def dl_model():
    # Load pre-trained VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Define input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1: Global Average Pooling
    pool_layer = GlobalAveragePooling2D()(input_layer)
    fc1 = Flatten()(pool_layer)
    fc2 = Dense(256, activation='relu')(fc1)
    fc3 = Dense(256, activation='relu')(fc2)

    # Block 2: Convolutional layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(fc3)
    conv2 = Conv2D(64, (3, 3), activation='relu')(conv1)
    pool2 = MaxPooling2D((2, 2))(conv2)

    # Add branch from Block 1
    branch = Dense(256, activation='relu')(fc3)
    branch = Dense(256, activation='relu')(branch)
    branch = Dense(128, activation='relu')(branch)

    # Combine outputs
    combined = Add()([branch, pool2])

    # Output layer
    output = Dense(10, activation='softmax')(combined)

    # Define model
    model = Model(inputs=input_layer, outputs=output)

    # Compile model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model