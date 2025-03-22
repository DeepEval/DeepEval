import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Convolutional layer, Batch Normalization, ReLU activation
    block1_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(input_layer)
    block1_bn = BatchNormalization()(block1_conv)
    block1_relu = ReLU()(block1_bn)
    
    # Block 2: Convolutional layer, Batch Normalization, ReLU activation
    block2_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    block2_bn = BatchNormalization()(block2_conv)
    block2_relu = ReLU()(block2_bn)
    
    # Block 3: Convolutional layer, Batch Normalization, ReLU activation
    block3_conv = Conv2D(filters=128, kernel_size=(3, 3), padding='same')(input_layer)
    block3_bn = BatchNormalization()(block3_conv)
    block3_relu = ReLU()(block3_bn)
    
    # Parallel branch
    branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same')(input_layer)
    branch_relu1 = ReLU()(branch_conv1)
    branch_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(input_layer)
    branch_relu2 = ReLU()(branch_conv2)
    
    # Add all paths
    add_layer = Add()([block1_relu, block2_relu, block3_relu, branch_relu1, branch_relu2])
    
    # Fully connected layers for classification
    flatten = Flatten()(add_layer)
    dense1 = Dense(units=1024, activation='relu')(flatten)
    dense2 = Dense(units=512, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Display the model summary
model.summary()