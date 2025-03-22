import keras
from keras.layers import Input, Conv2D, Add, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path branches
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    branch2 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch2)
    
    branch3 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(branch3)
    
    # Concatenate outputs of the branches
    concatenated = keras.layers.concatenate([branch1, branch2, branch3], axis=-1)
    
    # 1x1 convolution to adjust the output dimensions
    adjusted = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(concatenated)
    
    # Add the adjusted branch to the original input branch
    added = Add()([adjusted, input_layer])
    
    # Batch normalization
    batch_norm = BatchNormalization()(added)
    
    # Flatten the result
    flattened = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model