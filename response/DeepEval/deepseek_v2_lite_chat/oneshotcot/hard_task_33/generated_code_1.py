import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, SeparableConv2D, Add, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    def branch(input_tensor):
        # 1x1 convolutional layer to increase depth
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 depthwise separable convolutional layer
        depthwise = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same')(conv1x1)
        # 1x1 convolutional layer to reduce depth
        conv1x1_2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(depthwise)
        # Add the branches to form the output
        add_branch = Add()([input_tensor, conv1x1_2])
        return add_branch
    
    # Apply the block to each branch
    branch1 = branch(input_layer)
    branch2 = branch(input_layer)
    branch3 = branch(input_layer)
    
    # Concatenate the outputs of the three branches
    concat = Concatenate(axis=-1)([branch1, branch2, branch3])
    
    # Flatten the concatenated tensor
    flatten = Flatten()(concat)
    
    # Add fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Build and print the model
model = dl_model()
model.summary()