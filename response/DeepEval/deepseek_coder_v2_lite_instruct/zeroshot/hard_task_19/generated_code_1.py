import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Multiply, Add

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Branch path
    branch = GlobalAveragePooling2D()(x)
    branch = Dense(128, activation='relu')(branch)
    branch = Dense(64, activation='relu')(branch)
    branch_weights = Dense(x.shape[1] * x.shape[2] * x.shape[3], activation='softmax')(branch)
    branch_weights = tf.reshape(branch_weights, (x.shape[0], x.shape[1], x.shape[2], x.shape[3]))
    
    # Apply weights to the input
    weighted_input = Multiply()([x, branch_weights])
    
    # Add the main path and the weighted input
    added_output = Add()([x, weighted_input])
    
    # Additional fully connected layers for classification
    z = GlobalAveragePooling2D()(added_output)
    z = Dense(128, activation='relu')(z)
    outputs = Dense(10, activation='softmax')(z)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Example usage
model = dl_model()
model.summary()