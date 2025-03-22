from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import Adam

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Define the model using the Functional API
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Convolutional layer with 1x1 max pooling
    conv1 = Conv2D(32, (1, 1), padding='same', activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(conv1)
    
    # Convolutional layer with 2x2 max pooling
    conv2 = Conv2D(64, (2, 2), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2)
    
    # Convolutional layer with 4x4 max pooling
    conv3 = Conv2D(128, (4, 4), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(conv3)
    
    # Flatten the output for concatenation
    flat1 = Flatten()(pool3)
    
    # Concatenate all the extracted features
    concat = concatenate([flat1, pool1.output, pool2.output, pool3.output])
    
    # Fully connected layer 1
    dense1 = Dense(256, activation='relu')(concat)
    
    # Fully connected layer 2
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=dense2)
    
    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and return the model
model = dl_model()
model.summary()