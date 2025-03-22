import keras
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Conv2DTranspose, concatenate
from keras.layers import Layer

# Number of convolutional filters to use
num_filters = 32
# Size of the convolutional kernels
kernel_size = (3, 3)
# Size of the max pooling window
pool_size = (2, 2)
# Dropout rate
dropout_rate = 0.5

# Input shape
input_shape = (32, 32, 3)
# Number of classes
num_classes = 10

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

def dl_model():
    # Input layers
    input_layer = Input(shape=input_shape)
    
    # First part: feature extraction
    x = Conv2D(num_filters, kernel_size, padding='same')(input_layer)
    x = MaxPooling2D(pool_size)(x)
    x = Dropout(rate=dropout_rate)(x)
    
    # Second part: enhancement of generalization
    x = Conv2D(num_filters*2, kernel_size, padding='same')(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Conv2D(num_filters*2, kernel_size, padding='same')(x)
    
    # Skip connections from the first part
    skip_connection = Conv2D(num_filters*2, kernel_size, padding='same', name='skip_connection')(x)
    
    # Third part: upsampling and restoration of spatial information
    x = concatenate([x, skip_connection], axis=-1)
    x = UpSampling2D(size=pool_size)(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    x = UpSampling2D(size=pool_size)(x)
    x = Conv2D(num_filters, kernel_size, padding='same')(x)
    
    # Output layer
    output_layer = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    # Model building
    model = Model(inputs=input_layer, outputs=output_layer)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()