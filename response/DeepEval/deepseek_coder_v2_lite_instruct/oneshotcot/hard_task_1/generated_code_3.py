import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Concatenate, Multiply, Activation
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Path 1: Global Average Pooling followed by two fully connected layers
    gap = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=64, activation='relu')(gap)
    dense2 = Dense(units=32, activation='relu')(dense1)
    path1_output = Dense(units=3, activation='sigmoid')(dense2)
    
    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp = GlobalMaxPooling2D()(conv1)
    dense1_gmp = Dense(units=64, activation='relu')(gmp)
    dense2_gmp = Dense(units=32, activation='relu')(dense1_gmp)
    path2_output = Dense(units=3, activation='sigmoid')(dense2_gmp)
    
    # Channel attention weights
    attention_weights = Add()([path1_output, path2_output])
    attention_weights = Activation('sigmoid')(attention_weights)
    
    # Apply attention weights to the original features
    conv1_weighted = Multiply()([conv1, attention_weights])
    
    # Block 2
    # Extract spatial features by separately applying average and max pooling
    avg_pool = GlobalAveragePooling2D()(conv1_weighted)
    max_pool = GlobalMaxPooling2D()(conv1_weighted)
    
    # Concatenate along the channel dimension
    concat_pool = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # 1x1 convolution and sigmoid activation
    conv_1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='sigmoid')(concat_pool)
    
    # Normalize the features by element-wise multiplication with Block 1 features
    conv1_weighted_normalized = Multiply()([conv1_weighted, conv_1x1])
    
    # Additional branch with 1x1 convolutional layer
    conv_1x1_branch = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(conv1_weighted_normalized)
    
    # Add the main path and the branch
    final_output = Add()([conv1_weighted_normalized, conv_1x1_branch])
    final_output = Activation('relu')(final_output)
    
    # Fully connected layer for final classification
    output_layer = Dense(units=10, activation='softmax')(final_output)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model