from tensorflow.keras.layers import Input, Dense, Reshape, Permute, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Constants
    height, width, channels = 32, 32, 3
    num_classes = 10
    groups = 3
    channels_per_group = channels // groups
    
    # Input layer
    input_layer = Input(shape=(height, width, channels))

    # Reshape the input tensor to (height, width, groups, channels_per_group)
    reshaped = Reshape((height, width, groups, channels_per_group))(input_layer)

    # Swap the third and fourth dimensions for channel shuffling
    shuffled = Permute((1, 2, 4, 3))(reshaped)

    # Reshape back to original input shape
    reshaped_back = Reshape((height, width, channels))(shuffled)

    # Flatten the tensor for the fully connected layer
    flattened = Flatten()(reshaped_back)

    # Fully connected layer with softmax activation for classification
    output_layer = Dense(num_classes, activation='softmax')(flattened)

    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# Example of how to use the model with CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = dl_model()
model.summary()