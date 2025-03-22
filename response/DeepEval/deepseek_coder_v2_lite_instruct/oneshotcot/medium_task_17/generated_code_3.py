import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    input_shape = (32, 32, 3)  # CIFAR-10 images are 32x32 with 3 channels (RGB)
    
    # Input layer
    input_layer = Input(shape=input_shape)
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped = Reshape((input_shape[0], input_shape[1], 3, int(input_shape[2]/3)))(input_layer)
    
    # Permute the dimensions to swap the third and fourth dimensions
    permuted = Permute((1, 2, 4, 3))(reshaped)
    
    # Reshape back to the original input shape
    reshaped_back = Reshape(input_shape)(permuted)
    
    # Conv2D layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_back)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Second Conv2D layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    
    # Flatten layer
    flatten_layer = Flatten()(pool2)
    
    # Fully connected layer
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()