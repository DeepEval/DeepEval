import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalAveragePooling2D



def dl_model():
    
    # Load CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize the pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Number of classes
    num_classes = 10

    # Model inputs
    input_main = Input(shape=(32, 32, 3))
    input_branch = Input(shape=(32, 32, 3))

    # Main path
    x = Conv2D(32, (3, 3), activation='relu')(input_main)
    x = MaxPooling2D()(x)
    x = GlobalAveragePooling2D()(x)

    # Add a fully connected layer
    x = Dense(512, activation='relu')(x)

    # Second fully connected layer
    x = Dense(256, activation='relu')(x)

    # Reshape output for multiplication
    x = Dense(32 * 32 * 3)(x)
    x = Reshape((32, 32, 3))(x)

    # Branch path
    y = input_branch

    # Combine outputs from both paths
    combined = concatenate([x, y])

    # Additional fully connected layers
    z = Dense(128, activation='relu')(combined)
    output = Dense(num_classes, activation='softmax')(z)

    # Create the model
    model = Model(inputs=[input_main, input_branch], outputs=output)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Print model summary
    model.summary()

    return model