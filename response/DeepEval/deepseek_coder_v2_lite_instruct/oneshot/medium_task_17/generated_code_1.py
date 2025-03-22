import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Reshape, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    input_layer = Input(shape=input_shape)
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped_layer = Reshape((32, 32, 3, 1)(input_layer)
    
    # Permute the dimensions to swap the third and fourth dimensions
    permuted_layer = Permute((1, 2, 4, 3))(reshaped_layer)
    
    # Reshape back to the original input shape
    reshaped_back_layer = Reshape((32, 32, 3))(permuted_layer)
    
    # Add convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(reshaped_back_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Flatten the output
    flatten_layer = Flatten()(pool3)
    
    # Pass through fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model