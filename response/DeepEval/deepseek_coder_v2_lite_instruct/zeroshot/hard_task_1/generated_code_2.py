import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Add, Activation, Multiply

def dl_model():
    input_shape = (32, 32, 3)
    
    # Block 1: Initial Convolutional Layer
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    
    # Path 1: Global Average Pooling followed by two fully connected layers
    gap1 = GlobalAveragePooling2D()(x)
    fc1 = Dense(64, activation='relu')(gap1)
    fc2 = Dense(64, activation='relu')(fc1)
    
    # Path 2: Global Max Pooling followed by two fully connected layers
    gmp1 = GlobalMaxPooling2D()(x)
    fc3 = Dense(64, activation='relu')(gmp1)
    fc4 = Dense(64, activation='relu')(fc3)
    
    # Combine outputs from both paths
    combined = Add()([fc2, fc4])
    attention_weights = Activation('sigmoid')(combined)
    attention_applied = Multiply()([x, attention_weights])
    
    # Block 2: Extract spatial features
    avg_pool = AveragePooling2D((2, 2))(attention_applied)
    max_pool = MaxPooling2D((2, 2))(attention_applied)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    conv1x1 = Conv2D(32, (1, 1), activation='relu')(concat)
    sigmoid_activation = Activation('sigmoid')(conv1x1)
    normalized_features = Multiply()([attention_applied, sigmoid_activation])
    
    # Ensure output channels match input channels
    final_branch = Conv2D(32, (1, 1))(normalized_features)
    output = Add()([normalized_features, final_branch])
    output = Activation('relu')(output)
    
    # Final classification layer
    flatten = Flatten()(output)
    dense = Dense(10, activation='softmax')(flatten)
    
    model = Model(inputs=inputs, outputs=dense)
    
    return model

# Example usage:
# model = dl_model()
# model.summary()