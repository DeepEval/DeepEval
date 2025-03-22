import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Normalize the images
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Add a convolutional layer
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    
    # Add three average pooling layers with different pool sizes and strides
    pool_1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(x)
    pool_2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    pool_3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(x)
    
    # Flatten the output of each pooling layer
    flat_1 = Flatten()(pool_1)
    flat_2 = Flatten()(pool_2)
    flat_3 = Flatten()(pool_3)
    
    # Concatenate the flattened layers
    concatenated = Concatenate()([flat_1, flat_2, flat_3])
    
    # Add fully connected layers
    fc_1 = Dense(128, activation='relu')(concatenated)
    fc_2 = Dense(64, activation='relu')(fc_1)
    
    # Output layer for 10 classes
    output_layer = Dense(10, activation='softmax')(fc_2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()