import keras
from keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Reshape, Multiply, Add, Flatten
from keras.models import Model

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Block 1
    avg_pool = GlobalAveragePooling2D()(input_layer)
    dense1 = Dense(units=32, activation='relu')(avg_pool)  # First fully connected layer
    dense2 = Dense(units=32, activation='relu')(dense1)    # Second fully connected layer
    reshape_layer = Reshape((1, 1, 32))(dense2)           # Reshape to match input layer's spatial dimensions
    weighted_features = Multiply()([input_layer, reshape_layer])  # Element-wise multiplication with the input
    
    # Block 2
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)  # First Conv Layer
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)  # Second Conv Layer
    max_pool = MaxPooling2D(pool_size=(2, 2))(conv2)  # Max pooling layer
    
    # Branching from Block 1 to Block 2
    block_output = Add()([weighted_features, max_pool])  # Adding the outputs of Block 1 and Block 2
    
    # Classification Layers
    flatten_layer = Flatten()(block_output)  # Flatten the combined output
    final_dense1 = Dense(units=128, activation='relu')(flatten_layer)  # First fully connected layer for classification
    final_output = Dense(units=10, activation='softmax')(final_dense1)  # Final output layer for 10 classes
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=final_output)

    return model