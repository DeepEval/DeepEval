from keras.datasets import cifar10
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose, concatenate, Dense, Flatten
from keras.layers import BatchNormalization, LeakyReLU, Activation

def dl_model():
    # Input shape
    input_shape = (32, 32, 3)
    
    # Loading the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize the inputs from integers to floats
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Encoder layers
    x = inputs
    x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = MaxPooling2D(pool_size=2)(x)
    
    # Decoder layers
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = concatenate([x, x])
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = concatenate([x, x])
    
    # Classification layers
    x = Flatten()(x)
    x = Dense(512)(x)
    x = LeakyReLU(alpha=0.01)(x)
    x = Dense(2)(x)
    outputs = Activation('softmax')(x)
    
    # Construct the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Instantiate the model
model = dl_model()