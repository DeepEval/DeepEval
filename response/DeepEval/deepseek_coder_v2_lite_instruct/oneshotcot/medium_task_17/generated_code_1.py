import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute, Reshape
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape the input tensor into (height, width, groups, channels_per_group)
    reshaped_layer = Reshape(target_shape=(32, 32, 3, 1))(input_layer)
    
    # Permute the dimensions to swap the third and fourth dimensions
    permuted_layer = Permute((1, 2, 4, 3))(reshaped_layer)
    
    # Reshape back to the original input shape
    original_shape_layer = Reshape(target_shape=(32, 32, 3))(permuted_layer)
    
    # Add convolutional layer followed by a pooling layer
    conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(original_shape_layer)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv)
    
    # Define a block with multiple convolutional layers and a max pooling layer
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output = block(input_tensor=max_pooling)
    batch_norm = BatchNormalization()(block_output)
    flatten_layer = Flatten()(batch_norm)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model