import keras
from keras.layers import Input, Conv2D, Add, GlobalAveragePooling2D, Dense, Multiply
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # First branch: 3x3 convolution
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Second branch: 5x5 convolution
    branch2 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Add the outputs of the two branches
    added = Add()([branch1, branch2])
    
    # Global Average Pooling layer
    global_avg_pooling = GlobalAveragePooling2D()(added)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(global_avg_pooling)
    dense2 = Dense(units=64, activation='relu')(dense1)
    
    # Softmax layer to generate attention weights
    attention_weights = Dense(units=10, activation='softmax')(dense2)
    
    # Multiply the attention weights with the global average pooled output
    weighted_output = Multiply()([global_avg_pooling, attention_weights])
    
    # Final output layer to get the probability distribution across 10 classes
    output_layer = Dense(units=10, activation='softmax')(weighted_output)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model