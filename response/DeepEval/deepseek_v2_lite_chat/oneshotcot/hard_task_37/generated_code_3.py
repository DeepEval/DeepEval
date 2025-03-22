import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Branch 1
    branch1_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch1_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch1_conv)
    
    # Block for Branch 1
    def block(input_tensor):
        path1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        path3 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3])
        return output_tensor
    
    branch1_block_output = block(branch1_pooling)
    branch1_batch_norm = BatchNormalization()(branch1_block_output)
    branch1_flatten = Flatten()(branch1_batch_norm)
    
    # Dense layers for Branch 1
    branch1_dense1 = Dense(units=128, activation='relu')(branch1_flatten)
    branch1_dense2 = Dense(units=64, activation='relu')(branch1_dense1)
    branch1_output = Dense(units=10, activation='softmax')(branch1_dense2)
    
    # Branch 2
    branch2_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch2_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(branch2_conv)
    
    branch2_block_output = block(branch2_pooling)
    branch2_batch_norm = BatchNormalization()(branch2_block_output)
    branch2_flatten = Flatten()(branch2_batch_norm)
    
    # Dense layers for Branch 2
    branch2_dense1 = Dense(units=128, activation='relu')(branch2_flatten)
    branch2_dense2 = Dense(units=64, activation='relu')(branch2_dense1)
    branch2_output = Dense(units=10, activation='softmax')(branch2_dense2)
    
    # Concatenate outputs from both branches
    combined_output = Concatenate()([branch1_output, branch2_output])
    
    # Final dense layers
    final_dense1 = Dense(units=256, activation='relu')(combined_output)
    final_dense2 = Dense(units=128, activation='relu')(final_dense1)
    output_layer = Dense(units=10, activation='softmax')(final_dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()