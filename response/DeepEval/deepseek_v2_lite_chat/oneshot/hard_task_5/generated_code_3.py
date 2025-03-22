import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Reshape, Permute, DepthwiseConv2D, Add, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups
    group_1, group_2, group_3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # Process each group through a 1x1 convolutional layer
    conv1_1 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(group_1)
    conv1_2 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(group_2)
    conv1_3 = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='relu')(group_3)
    
    # Concatenate the outputs of these three groups
    fused_features = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
    
    # Block 1 (similar to the original example)
    block1_output = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(fused_features)
    block1_output = MaxPooling2D(pool_size=(2, 2))(block1_output)
    
    # Block 2 (channel shuffling and reshaping)
    block2_output = Reshape((block1_output.shape[1], block1_output.shape[2], 3))(block1_output)
    group_1, group_2, group_3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(block2_output)
    
    # Swap dimensions
    group_1 = Permute((3, 1, 2))(group_1)
    group_2 = Permute((3, 1, 2))(group_2)
    group_3 = Permute((3, 1, 2))(group_3)
    
    # Reshape and reshape back
    group_1 = Reshape((block2_output.shape[1], block2_output.shape[2], 3))(group_1)
    group_2 = Reshape((block2_output.shape[1], block2_output.shape[2], 3))(group_2)
    group_3 = Reshape((block2_output.shape[1], block2_output.shape[2], 3))(group_3)
    
    # Process with depthwise convolution
    conv2_1 = DepthwiseConv2D((3, 3), padding='same')(group_1)
    conv2_2 = DepthwiseConv2D((3, 3), padding='same')(group_2)
    conv2_3 = DepthwiseConv2D((3, 3), padding='same')(group_3)
    
    # Add branch from input directly
    branch_output = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    
    # Combine outputs from main path and branch
    combined_output = Add()([block2_output, branch_output])
    
    # Flatten and fully connected layers
    flat1 = Flatten()(combined_output)
    dense1 = Dense(units=256, activation='relu')(flat1)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])