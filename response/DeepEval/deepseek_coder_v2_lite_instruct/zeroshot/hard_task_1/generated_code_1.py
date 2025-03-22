import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Add, Activation, Multiply, Concatenate, AveragePooling2D, MaxPooling2D, Lambda
from tensorflow.keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(input_layer)
    
    # Path 1: Global Average Pooling followed by two fully connected layers
    gap1 = GlobalAveragePooling2D()(x)
    dense1 = Dense(64, activation='relu')(gap1)
    dense2 = Dense(64, activation='relu')(dense1)
    
    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp1 = GlobalMaxPooling2D()(x)
    dense3 = Dense(64, activation='relu')(gmp1)
    dense4 = Dense(64, activation='relu')(dense3)
    
    # Add outputs from both paths
    added_features = Add()([dense2, dense4])
    
    # Generate channel attention weights
    attention_weights = Activation('sigmoid')(added_features)
    
    # Apply attention weights to the original features
    channel_attention_features = Multiply()([x, attention_weights])
    
    # Block 2: Extract spatial features
    avg_pool = AveragePooling2D((3, 3))(channel_attention_features)
    max_pool = MaxPooling2D((3, 3))(channel_attention_features)
    
    # Concatenate the outputs of average and max pooling
    concat_features = Concatenate(axis=-1)([avg_pool, max_pool])
    
    # Apply a 1x1 convolution and sigmoid activation
    conv1x1 = Conv2D(1, (1, 1), activation='sigmoid')(concat_features)
    normalized_features = Multiply()([channel_attention_features, conv1x1])
    
    # Additional branch to ensure output channels align with input channels
    final_branch = Conv2D(3, (1, 1))(normalized_features)
    
    # Add the final branch to the main path
    output_layer = Add()([final_branch, normalized_features])
    output_layer = Activation('relu')(output_layer)
    
    # Final classification layer
    final_output = Dense(10, activation='softmax')(output_layer)
    
    # Define and compile the model
    model = Model(inputs=input_layer, outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()