import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Add, Multiply, Activation, Lambda
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Block 1
    def block1(x):
        # Path 1: Global Average Pooling followed by two fully connected layers
        gap = GlobalAveragePooling2D()(x)
        dense1 = Dense(units=64, activation='relu')(gap)
        dense2 = Dense(units=32, activation='relu')(dense1)
        
        # Path 2: Global Max Pooling followed by two fully connected layers
        gmp = GlobalMaxPooling2D()(x)
        dense_gmp1 = Dense(units=64, activation='relu')(gmp)
        dense_gmp2 = Dense(units=32, activation='relu')(dense_gmp1)
        
        # Concatenate the outputs of both paths
        concat = Concatenate()([dense2, dense_gmp2])
        
        # Generate channel attention weights
        attention_weights = Dense(units=x.shape[-1], activation='sigmoid')(concat)
        
        # Apply attention weights to the original features
        output = Multiply()([x, attention_weights])
        
        return output
    
    block1_output = block1(conv1)
    
    # Block 2: Extract spatial features
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=[1, 2], keepdims=True))(block1_output)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=[1, 2], keepdims=True))(block1_output)
    spatial_features = Concatenate(axis=-1)([avg_pool, max_pool])
    spatial_features = Conv2D(filters=block1_output.shape[-1], kernel_size=(1, 1), activation='sigmoid')(spatial_features)
    
    # Multiply spatial features with channel-wise features
    normalized_features = Multiply()([block1_output, spatial_features])
    
    # Additional branch to ensure output channels align with input channels
    final_branch = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(normalized_features)
    
    # Add the main path and the final branch
    final_output = Add()([normalized_features, final_branch])
    
    # Flatten and pass through fully connected layers for final classification
    flatten = Flatten()(final_output)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()