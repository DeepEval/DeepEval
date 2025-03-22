import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

def dl_model():
    
    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Define the convolutional layer 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Define the convolutional layer 2
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # Define the max-pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Add the input layer with the max-pooling layer
    added_output = keras.layers.Add()([input_layer, max_pooling])
    
    # Flatten the added output
    flatten_layer = Flatten()(added_output)
    
    # Define the first fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    
    # Define the second fully connected layer
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model