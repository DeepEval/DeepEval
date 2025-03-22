import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam

def dl_model():
    # Constants
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    IMAGE_CHANNELS = 3
    EPOCHS = 20
    BATCH_SIZE = 64
    OPTIMIZER = Adam(lr=0.001)

    # Input shape
    input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
    
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Add a dimension to the input
    input_layer = Input(shape=input_shape)
    
    # Stage 1: Convolution and Max Pooling
    x = Conv2D(32, (3, 3), padding='same')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 2: Convolution and Max Pooling
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 3: Convolution and Max Pooling
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Stage 4: Skip Connection
    x1 = Conv2D(64, (3, 3), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Concatenate()([x, x1])
    
    # Stage 5: Dropout and Convolution
    x = Dropout(0.2)(x)
    x = Conv2D(128, (3, 3), padding='same')(x)
    
    # Stage 6: Skip Connection
    x1 = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = Concatenate()([x, x1])
    
    # Stage 7: Dropout and Convolution
    x = Dropout(0.2)(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    
    # Stage 8: Skip Connection
    x1 = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = Concatenate()([x, x1])
    
    # Stage 9: Final Convolution
    output = Conv2D(10, (1, 1), padding='softmax')(x)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output)
    
    # Compile the model
    model.compile(optimizer=OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build the model
model = dl_model()