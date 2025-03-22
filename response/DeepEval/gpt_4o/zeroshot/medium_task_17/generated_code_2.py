import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Permute, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = x_train.shape[1:]  # (32, 32, 3)
    
    # Normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Reshape the input tensor to (height, width, groups, channels_per_group)
    channels = input_shape[2]
    groups = 3
    channels_per_group = channels // groups
    
    reshaped = Reshape((input_shape[0], input_shape[1], groups, channels_per_group))(inputs)
    
    # Swap the third and fourth dimensions to enable channel shuffling
    shuffled = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to original input shape
    reshaped_back = Reshape(input_shape)(shuffled)
    
    # Flatten the output
    flatten = Flatten()(reshaped_back)
    
    # Add a fully connected layer with softmax activation
    outputs = Dense(10, activation='softmax')(flatten)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()