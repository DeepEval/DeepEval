from keras.applications import VGG16
from keras.layers import Input, Dense, Dropout, Flatten
from keras.models import Model

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False


def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Add the 1x1 convolution branch
    x = Conv2D(64, (1, 1), activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Add the 1x1 convolution followed by 3x3 convolution branch
    x = Conv2D(64, (1, 1), activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Add the 1x1 convolution followed by two consecutive 3x3 convolutions branch
    x = Conv2D(64, (1, 1), activation='relu')(input_layer)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Add the average pooling followed by 1x1 convolution branch
    x = AveragePooling2D((2, 2))(input_layer)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # Concatenate the outputs from all branches
    x = Concatenate()([x, x, x, x])

    # Add the final fully connected layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

    return model