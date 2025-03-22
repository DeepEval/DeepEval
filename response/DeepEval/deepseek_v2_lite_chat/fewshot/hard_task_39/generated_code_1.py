import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block: Process with average pooling layers of different scales
    def block_1(input_tensor):
        avg_pool1 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        avg_pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        avg_pool3 = AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        flat_output = Flatten()(Concatenate()([avg_pool1, avg_pool2, avg_pool3]))
        dense1 = Dense(units=128, activation='relu')(flat_output)
        return dense1
    
    # Second block: Process with depthwise separable convolutions in parallel
    def block_2(input_tensor):
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        pool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, pool])
        bn = BatchNormalization()(output_tensor)
        flat = Flatten()(bn)
        dense1 = Dense(units=128, activation='relu')(flat)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Connect the blocks
    block1_output = block_1(input_tensor=input_layer)
    output_layer = block_2(input_tensor=block1_output)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.summary()