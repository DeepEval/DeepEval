import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, Activation, AveragePooling2D, MaxPooling2D, Multiply, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_layer)
    
    # Channel attention paths
    # Path 1: Global Average Pooling -> Dense Layers
    gap1 = GlobalAveragePooling2D()(conv1)
    dense1_1 = Dense(units=128, activation='relu')(gap1)
    dense1_2 = Dense(units=64, activation='relu')(dense1_1)
    
    # Path 2: Global Max Pooling -> Dense Layers
    gmp1 = GlobalMaxPooling2D()(conv1)
    dense2_1 = Dense(units=128, activation='relu')(gmp1)
    dense2_2 = Dense(units=64, activation='relu')(dense2_1)
    
    # Concatenate channel attention paths
    concat = Concatenate()([dense1_2, dense2_2])
    
    # Activation for channel attention weights
    attention_weights = Activation('sigmoid')(concat)
    
    # Element-wise multiplication with original features for channel attention
    attention_features = Multiply()([conv1, attention_weights])
    
    # Spatial feature extraction paths
    # Average pooling
    avg_pool1 = AveragePooling2D(pool_size=(2, 2))(attention_features)
    avg_pool2 = AveragePooling2D(pool_size=(2, 2))(attention_features)
    
    # Max pooling
    max_pool1 = MaxPooling2D(pool_size=(2, 2))(attention_features)
    max_pool2 = MaxPooling2D(pool_size=(2, 2))(attention_features)
    
    # Concatenate average and max pooling features
    concat_spatial = Concatenate(axis=1)([avg_pool1, avg_pool2, max_pool1, max_pool2])
    
    # Fully connected layer for final classification
    fc1 = Dense(units=512, activation='relu')(concat_spatial)
    output_layer = Dense(units=10, activation='softmax')(fc1)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model