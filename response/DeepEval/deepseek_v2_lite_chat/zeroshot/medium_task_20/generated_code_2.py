import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from keras.optimizers import Adam

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Function to define the model
def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First path: 1x1 convolution
    path1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    
    # Second path: 3x3 convolution, followed by 1x1 convolution
    path2 = Conv2D(64, (3, 3), activation='relu')(path1)
    path2 = Conv2D(64, (1, 1), activation='relu')(path2)
    
    # Third path: single 3x3 convolution, followed by 1x1 convolution
    path3 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    path3 = Conv2D(64, (1, 1), activation='relu')(path3)
    
    # Fourth path: max pooling, followed by 1x1 convolution
    path4 = MaxPooling2D(pool_size=(2, 2))(input_layer)
    path4 = Conv2D(64, (1, 1), activation='relu')(path4)
    
    # Concatenate outputs from each path
    concatenated = concatenate([path2, path3, path4])
    
    # Flatten and pass through a dense layer
    dense = Flatten()(concatenated)
    dense = Dense(128, activation='relu')(dense)
    
    # Output layer with softmax activation
    output_layer = Dense(10, activation='softmax')(dense)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()