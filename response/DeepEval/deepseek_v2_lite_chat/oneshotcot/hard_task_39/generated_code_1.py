import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1: Multiple max pooling layers
    def block1(input_tensor):
        # 1x1 max pooling
        pool1x1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        # 2x2 max pooling
        pool2x2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        # 4x4 max pooling
        pool4x4 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        # Flatten each pooling result and concatenate
        flat1 = Flatten()(pool1x1)
        flat2 = Flatten()(pool2x2)
        flat3 = Flatten()(pool4x4)
        concat = Concatenate()([flat1, flat2, flat3])
        
        return concat
    
    # Block 2: Multiple branches with convolutions and max pooling
    def block2(input_tensor):
        # 1x1 convolution
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 convolution
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 5x5 convolution
        conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 max pooling
        pool3x3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='valid')(input_tensor)
        # Concatenate outputs from all branches
        concat_branch1 = Concatenate()([conv1x1, conv3x3, conv5x5, pool3x3])
        
        return concat_branch1
    
    # First block output
    block1_output = block1(input_layer)
    
    # Convert output of Block 1 to 4D tensor for input into Block 2
    reshape = Reshape((-1, 16))(block1_output)  # Adjust dimensions as needed
    
    # Second block input
    block2_input = block2(reshape)
    
    # Concatenate outputs from both blocks
    concat_model_output = Concatenate()([block1_output, block2_input])
    
    # Add fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat_model_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()