import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy

# Load CIFAR-10 data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model parameters
input_shape = (32, 32, 3)  # Input image shape
num_classes = 10  # Number of classes

# Function to create the model
def dl_model():
    # Input layer
    inputs = Input(shape=input_shape)
    
    # 1x1 convolutional layer
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    
    # Branch 1: Local features
    x1 = Conv2D(64, (3, 3), activation='relu')(x)
    
    # Branch 2: Downsample and upsample
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D(size=2)(x)
    
    # Branch 3: Downsample and upsample
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = UpSampling2D(size=2)(x)
    
    # Concatenate all branches
    x = Concatenate()([x1, x])
    
    # Final 1x1 convolutional layer
    x = Conv2D(num_classes, (1, 1), activation='softmax')(x)
    
    # Model
    model = Model(inputs=inputs, outputs=x)
    
    # Compile the model
    model.compile(optimizer=Adam(),
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[SparseCategoricalAccuracy()])
    
    return model

# Build and return the model
model = dl_model()
model.summary()