import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolution
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # First block
    conv1_block = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    
    # Second block
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1_block)
    conv2_block = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    # Third block
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu')(conv2_block)
    conv3_block = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    
    # Add outputs of the blocks to enhance feature representation
    add_layer = Add()([conv1, conv1_block, conv2, conv2_block, conv3, conv3_block])
    
    # Flatten and pass through fully connected layers
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=256, activation='relu')(flatten)
    output = Dense(units=10, activation='softmax')(dense1)  # Assuming 10 classes for CIFAR-10
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()