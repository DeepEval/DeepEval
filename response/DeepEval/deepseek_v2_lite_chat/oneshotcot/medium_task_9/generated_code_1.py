import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Basic Block 1
    def block1(input_tensor):
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm
    
    # Branch connecting to the input
    input_branch = block1(input_layer)
    
    # Basic Block 2
    def block2(input_tensor):
        conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        batch_norm = BatchNormalization()(conv)
        return batch_norm
    
    # Extracting features through another convolutional layer
    conv_layer = Conv2D(filters=64, kernel_size=(1, 1))(input_layer)
    batch_norm_layer = BatchNormalization()(conv_layer)
    
    # Basic Block 1
    block1_output = block1(input_tensor=input_branch)
    # Basic Block 2
    block2_output = block2(input_tensor=input_branch)
    
    # Feature fusion
    fused_output = Concatenate(axis=-1)([block1_output, block2_output, conv_layer, batch_norm_layer])
    
    # Add average pooling layer to downsample the feature map
    avg_pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(fused_output)
    
    # Flatten the output for fully connected layers
    flattened = Flatten()(avg_pool)
    
    # Fully connected layer for classification
    dense = Dense(units=128, activation='relu')(flattened)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Build the model
model = dl_model()
model.summary()