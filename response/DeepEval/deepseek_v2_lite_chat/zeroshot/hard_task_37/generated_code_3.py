import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add, concatenate

# Number of convolutional filters
CONV_FILTERS = 32
# Kernel sizes for the convolutional layers
CONV_KERNEL_SIZE = (3, 3)
# Pool size for max pooling
POOL_SIZE = (2, 2)
# Flatten size
FLATTEN_SIZE = 64

def dl_model():
    # Input layer
    inputs = Input(shape=(28, 28, 1))
    
    # First branch
    branch1 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(inputs)
    branch1 = MaxPooling2D(pool_size=POOL_SIZE)(branch1)
    branch1 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(branch1)
    branch1 = MaxPooling2D(pool_size=POOL_SIZE)(branch1)
    branch1 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(branch1)
    branch1 = Flatten()(branch1)
    branch1 = Dense(FLATTEN_SIZE, activation='relu')(branch1)
    
    # Second branch
    branch2 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(inputs)
    branch2 = MaxPooling2D(pool_size=POOL_SIZE)(branch2)
    branch2 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(branch2)
    branch2 = MaxPooling2D(pool_size=POOL_SIZE)(branch2)
    branch2 = Conv2D(CONV_FILTERS, CONV_KERNEL_SIZE, activation='relu')(branch2)
    branch2 = Flatten()(branch2)
    branch2 = Dense(FLATTEN_SIZE, activation='relu')(branch2)
    
    # Combine branches
    combined = Add()([branch1, branch2])
    
    # Fully connected layers
    output = Dense(10, activation='softmax')(combined)
    
    # Model
    model = Model(inputs=inputs, outputs=output)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Assuming you want to create and evaluate the model
model = dl_model()
model.summary()