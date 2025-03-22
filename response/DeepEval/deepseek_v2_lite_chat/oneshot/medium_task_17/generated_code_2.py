import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Permute, Reshape, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Reshape to (height, width, groups, channels_per_group) where groups=3 and channels_per_group=channels
    groups = 3
    channels_per_group = 32 // groups
    reshaped_input = Permute((2, 3, 1))(input_layer)
    
    # Shuffle channels
    reshaped_input = Conv2D(filters=channels_per_group, kernel_size=(1, 1), padding='same')(reshaped_input)
    
    # Reshape back to original shape
    reshaped_input = Reshape((32 // channels_per_group, 32 // channels_per_group, groups, channels_per_group))(reshaped_input)
    
    # Convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(reshaped_input)
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(conv2)
    
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv3)
    
    # Parallel paths
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(path1)
        path3 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(path2)
        return Concatenate()(inputs=[path1, path2, path3])
    
    block_output = block(pool1)
    
    # Batch normalization and flattening
    bn = BatchNormalization()(block_output)
    flat = Flatten()(bn)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()