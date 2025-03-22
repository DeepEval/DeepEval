import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped_input = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    
    # Permute the dimensions to enable channel shuffling
    permuted_input = Permute((4, 1, 2, 3))(reshaped_input)
    
    # Reshape back to the original input shape
    reshaped_back_input = Reshape(target_shape=(32, 32, 3))(permuted_input)
    
    # Continue with the rest of the model
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(reshaped_back_input)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv2)
    conv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Flatten the output
    flatten_layer = Flatten()(pool3)
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model