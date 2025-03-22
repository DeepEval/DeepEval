import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Define the model input
    input_layer = Input(shape=(28, 28, 1))
    
    # Convolutional layer with 32 filters, 3x3 kernel size
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    # Convolutional layer with 64 filters, 3x3 kernel size
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    # Flatten layer
    flatten = Flatten()(pool2)
    
    # Two parallel paths for feature extraction
    path1 = Dense(128, activation='relu')(flatten)
    path2 = Dense(128, activation='relu')(flatten)
    
    # Concatenate the outputs of the two paths
    concat = Concatenate()([path1, path2])
    
    # Fully connected layers
    dense1 = Dense(256, activation='relu')(concat)
    dense1 = BatchNormalization()(dense1)
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = BatchNormalization()(dense2)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])