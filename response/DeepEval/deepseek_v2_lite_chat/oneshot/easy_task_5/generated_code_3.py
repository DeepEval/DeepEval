import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape the data
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
    
    # Normalize the data
    x_train /= 255
    x_test /= 255
    
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    # Model architecture
    input_layer = Input(shape=(28, 28, 1))
    
    # Dimensionality reduction with 1x1 convolution
    conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # 3x3 convolutional layer for feature extraction
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
    
    # 1x1 convolutional layer to restore dimensionality
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    
    # Flatten the feature map
    flatten = Flatten()(conv3)
    
    # Fully connected layer with 10 neurons for classification
    dense = Dense(units=10, activation='softmax')(flatten)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()