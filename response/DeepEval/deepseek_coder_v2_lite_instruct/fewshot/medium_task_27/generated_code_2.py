import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply, Concatenate

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels

    # First branch (3x3 convolutional layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1_1)
    
    # Second branch (5x5 convolutional layer)
    conv2_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2_2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2_1)
    
    # Adding the outputs of both branches
    added = Add()([conv1_2, conv2_2])
    
    # Global average pooling
    gap = GlobalAveragePooling2D()(added)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(gap)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Softmax for class probabilities
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Attention mechanism
    attention_weights = Dense(units=2, activation='softmax')(dense2)  # 2 weights for 2 branches
    attention_weights_expanded = keras.layers.Reshape((1, 1, 2))(attention_weights)
    
    # Multiply attention weights with branch outputs
    weighted_conv1_2 = Multiply()([conv1_2, attention_weights_expanded[:,:,:,:1]])
    weighted_conv2_2 = Multiply()([conv2_2, attention_weights_expanded[:,:,:,1:]])
    
    # Combine weighted outputs
    combined = Add()([weighted_conv1_2, weighted_conv2_2])
    
    # Final output layer
    final_output = Dense(units=10, activation='softmax')(combined)
    
    model = keras.Model(inputs=input_layer, outputs=final_output)
    
    return model