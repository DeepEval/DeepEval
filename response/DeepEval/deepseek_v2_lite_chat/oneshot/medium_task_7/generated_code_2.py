import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Load and prepare CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    batchnorm1 = BatchNormalization()(conv1)
    
    # Second convolutional layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batchnorm1)
    batchnorm2 = BatchNormalization()(conv2)
    
    # Third convolutional layer
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(batchnorm2)
    
    # Add all three convolutional layers and concatenate
    add = Add()([conv1, conv2, conv3])
    
    # Fourth convolutional layer
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(add)
    batchnorm3 = BatchNormalization()(conv4)
    
    # Fifth convolutional layer
    conv5 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(batchnorm3)
    
    # Concatenate the outputs of the first and fifth convolutional layers
    concat = Concatenate()([conv1, conv5])
    
    # Pooling layer
    pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(concat)
    
    # Flatten the output
    flatten = Flatten()(pool)
    
    # Two fully connected layers
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    
    # Output layer with 10 classes (CIFAR-10)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])