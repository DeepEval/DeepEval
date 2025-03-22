from tensorflow.keras.layers import Input, Conv2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Define the input
    input_shape = (32, 32, 3)
    inputs = Input(shape=input_shape)

    # Main pathway
    # Path 1: Simple 1x1 Convolution
    path1 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)

    # Path 2: Series of 1x1, 1x3, and 3x1 Convolutions
    path2_1 = Conv2D(32, (1, 1), padding='same', activation='relu')(inputs)
    path2_2 = Conv2D(32, (1, 3), padding='same', activation='relu')(path2_1)
    path2_3 = Conv2D(32, (3, 1), padding='same', activation='relu')(path2_2)

    # Concatenate the outputs from path1 and path2
    concatenated = Concatenate()([path1, path2_3])

    # Another 1x1 Convolution after concatenation
    main_output = Conv2D(3, (1, 1), padding='same', activation='relu')(concatenated)

    # Additive operation with the input (Skip connection)
    fused_output = Add()([main_output, inputs])

    # Flatten and fully connected layers for classification
    flat = Flatten()(fused_output)
    fc1 = Dense(128, activation='relu')(flat)
    classification_output = Dense(10, activation='softmax')(fc1)

    # Construct the model
    model = Model(inputs=inputs, outputs=classification_output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# Load CIFAR-10 data for demonstration
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Create and compile the model
model = dl_model()

# You can now train the model using model.fit() and evaluate using model.evaluate()