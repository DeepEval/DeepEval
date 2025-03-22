import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Flatten, Dense, Concatenate
from keras.layers.advanced_activations import LeakyReLU

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First convolutional layer with batch normalization and ReLU activation
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    activation1 = Activation('relu')(bn1)
    
    # Second convolutional layer with batch normalization and ReLU activation
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(activation1)
    bn2 = BatchNormalization()(conv2)
    activation2 = Activation('relu')(bn2)
    
    # Global average pooling
    avg_pool = MaxPooling2D(pool_size=(2, 2))(bn2)
    
    # Flatten and fully connected layers
    flat = Flatten()(avg_pool)
    dense1 = Dense(units=512, activation=LeakyReLU(alpha=0.01))(flat)
    dense2 = Dense(units=256, activation=LeakyReLU(alpha=0.01))(dense1)
    
    # Reshape to match the number of initial feature channels
    reshape = Dense(activation='sigmoid')(dense2)
    reshape.set_shape(initial_shape=(512,))
    
    # Weighted feature maps generation
    concat = Concatenate()([bn2, reshape])
    
    # Final convolutional layer and output
    conv3 = Conv2D(filters=3, kernel_size=(1, 1), padding='same')(concat)
    avg_pool_final = MaxPooling2D(pool_size=(2, 2))(conv3)
    dense3 = Dense(units=10, activation='softmax')(avg_pool_final)
    
    model = Model(inputs=input_layer, outputs=dense3)
    
    return model