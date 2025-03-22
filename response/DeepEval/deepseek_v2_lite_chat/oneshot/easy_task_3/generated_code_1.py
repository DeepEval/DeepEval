import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block of two convolutional layers followed by a max pooling layer
    def block1():
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        return pool1
    
    # Second block of two convolutional layers followed by two more convolutional layers
    def block2():
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
        conv4 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv3)
        conv5 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv4)
        conv6 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv5)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv6)
        return pool2
    
    # Third block of four parallel convolutional paths
    def block3():
        path1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_layer)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')(input_layer)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=1)(input_layer)
        output = Concatenate(axis=-1)([path1, path2, path3, path4])
        return output
    
    # Apply the blocks sequentially
    x1 = block1()
    x2 = block2()
    x3 = block3()
    
    # Combine the outputs of the three blocks
    combined = Concatenate()([x1, x2, x3])
    
    # Batch normalization and flattening
    batch_norm = BatchNormalization()(combined)
    flatten = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)  # Assuming 10 classes for MNIST
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])