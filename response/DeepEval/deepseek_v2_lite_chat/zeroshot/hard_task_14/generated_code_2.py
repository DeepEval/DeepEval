import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from keras.layers import BatchNormalization, Activation
from keras.utils import to_categorical

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Number of classes
num_classes = y_train.shape[1]

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(64, (3, 3), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    
    # Two fully connected layers for the main path
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Weights reshaping and multiplication with the original feature map
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Branch path
    branch_input = Conv2D(64, (3, 3), padding='same')(x)
    branch_input = BatchNormalization()(branch_input)
    branch_input = Activation('relu')(branch_input)
    
    # Multiply element-wise with the main path's weights
    x = concatenate([x, branch_input])
    
    # Another fully connected layer for the branch path
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Third fully connected layer for the branch path
    x = Dense(128)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Output layer
    output = Dense(num_classes, activation='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()

# Print model summary
model.summary()