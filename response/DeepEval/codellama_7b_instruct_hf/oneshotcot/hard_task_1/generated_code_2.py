import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Flatten, Activation, Add, Multiply, Concatenate, BatchNormalization, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Global Average Pooling and Fully Connected layers
    path1 = GlobalAveragePooling2D()(input_layer)
    path1 = Dense(units=64, activation='relu')(path1)
    path1 = Dense(units=32, activation='relu')(path1)
    
    # Block 1: Global Max Pooling and Fully Connected layers
    path2 = GlobalMaxPooling2D()(input_layer)
    path2 = Dense(units=64, activation='relu')(path2)
    path2 = Dense(units=32, activation='relu')(path2)
    
    # Block 1: Channel attention
    path1 = Dense(units=32, activation='softmax')(path1)
    path2 = Dense(units=32, activation='softmax')(path2)
    path1 = Multiply()([path1, input_layer])
    path2 = Multiply()([path2, input_layer])
    path1 = Activation('relu')(path1)
    path2 = Activation('relu')(path2)
    block1 = Add()([path1, path2])
    
    # Block 2: Spatial features
    path1 = GlobalAveragePooling2D()(block1)
    path2 = GlobalMaxPooling2D()(block1)
    path1 = Dense(units=16, activation='relu')(path1)
    path2 = Dense(units=16, activation='relu')(path2)
    path1 = Concatenate()([path1, path2])
    path1 = Dense(units=8, activation='relu')(path1)
    path1 = Dense(units=1, activation='sigmoid')(path1)
    block2 = Multiply()([path1, input_layer])
    
    # Final classification
    output_layer = Dense(units=10, activation='softmax')(block2)
    
    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model