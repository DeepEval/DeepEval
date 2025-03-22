import keras
from keras.layers import Input, Conv2D, ReLU, Add, Concatenate, Dense, Flatten
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (28, 28, 1)  # MNIST image shape
    input = Input(shape=input_shape)

    # Main path to enhance features
    def enhance_features(inputs):
        x = Conv2D(32, (3, 3), activation='relu')(inputs)  # Conv layer
        x = Conv2D(32, (3, 3), activation='relu')(x)  # Conv layer
        x = Flatten()(x)
        return x

    # Branch path with a simple convolutional layer
    def branch_features(inputs):
        x = Conv2D(32, (3, 3), padding='same')(inputs)  # Conv layer
        x = Flatten()(x)
        return x

    # Main path block
    main_path_output = enhance_features(input)
    for _ in range(3):
        main_path_output = enhance_features(main_path_output)

    # Branch path
    branch_output = branch_features(input)

    # Concatenate along the channel dimension
    fused_output = Concatenate(axis=-1)([main_path_output, branch_output])

    # Fully connected layer
    output = Dense(10, activation='softmax')(fused_output)

    # Model construction
    model = Model(inputs=input, outputs=output)

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

# Build and return the model
model = dl_model()