import keras
from keras.layers import Input, Conv2D, Add, Concatenate, Flatten, Dense
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main pathway: First branch with a 1x1 convolution
    branch1x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)

    # Main pathway: Second branch with 1x1, 1x3, and 3x1 convolutions
    branch1x3 = Conv2D(32, (1, 3), padding='same', activation='relu')(input_layer)
    branch3x1 = Conv2D(32, (3, 1), padding='same', activation='relu')(branch1x3)

    # Concatenate the outputs of the two branches
    concatenated_branches = Concatenate()([branch1x1, branch3x1])

    # Pass through another 1x1 convolution to produce main output
    main_output = Conv2D(3, (1, 1), padding='same', activation='relu')(concatenated_branches)

    # Additive connection from input directly to the main output
    fused_output = Add()([input_layer, main_output])

    # Flattening for fully connected layers
    flatten = Flatten()(fused_output)

    # Fully connected layers for classification
    dense1 = Dense(128, activation='relu')(flatten)
    output_layer = Dense(10, activation='softmax')(dense1)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example: Loading CIFAR-10 data and compiling the model
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()