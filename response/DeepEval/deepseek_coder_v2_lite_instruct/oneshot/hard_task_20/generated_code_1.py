import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main path
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(input_layer)
    
    # Merge outputs from main path
    merged_main = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Combine main and branch paths
    fused_features = keras.layers.add([merged_main, branch_conv])
    
    # Batch normalization
    batch_norm = BatchNormalization()(fused_features)
    
    # Flatten the output
    flatten_layer = Flatten()(batch_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and summarize the model
model = dl_model()
model.summary()