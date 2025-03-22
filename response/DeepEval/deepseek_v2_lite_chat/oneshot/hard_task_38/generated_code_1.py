import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape the data
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    
    # Normalize inputs from 0-255 to 0-1
    x_train /= 255
    x_test /= 255
    
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First pathway
    def pathway1():
        # Conv2D layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        # Batch normalization
        batch_norm1 = BatchNormalization()(conv)
        # MaxPooling2D layer
        max_pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
        # Repeat block three times
        for _ in range(3):
            conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(max_pool1)
            batch_norm1 = BatchNormalization()(conv)
            max_pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
        
        # Flatten the output
        flat1 = Flatten()(max_pool1)
        return flat1
    
    flat_output1 = pathway1()
    
    # Second pathway
    def pathway2():
        # Conv2D layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
        # Batch normalization
        batch_norm2 = BatchNormalization()(conv)
        # MaxPooling2D layer
        max_pool2 = MaxPooling2D(pool_size=(3, 3))(batch_norm2)
        # Repeat block three times
        for _ in range(3):
            conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(max_pool2)
            batch_norm2 = BatchNormalization()(conv)
            max_pool2 = MaxPooling2D(pool_size=(3, 3))(batch_norm2)
        
        # Flatten the output
        flat2 = Flatten()(max_pool2)
        return flat2
    
    flat_output2 = pathway2()
    
    # Concatenate outputs from both pathways
    concat = Concatenate()([flat_output1, flat_output2])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])