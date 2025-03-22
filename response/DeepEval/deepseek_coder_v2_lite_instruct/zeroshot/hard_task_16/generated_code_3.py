import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def split_and_extract(x):
        x1, x2, x3 = tf.split(x, 3, axis=-1)
        x1 = Conv2D(32, (1, 1), activation='relu')(x1)
        x1 = Conv2D(32, (3, 3), activation='relu')(x1)
        x1 = Conv2D(32, (1, 1), activation='relu')(x1)
        
        x2 = Conv2D(32, (1, 1), activation='relu')(x2)
        x2 = Conv2D(32, (3, 3), activation='relu')(x2)
        x2 = Conv2D(32, (1, 1), activation='relu')(x2)
        
        x3 = Conv2D(32, (1, 1), activation='relu')(x3)
        x3 = Conv2D(32, (3, 3), activation='relu')(x3)
        x3 = Conv2D(32, (1, 1), activation='relu')(x3)
        
        return tf.concat([x1, x2, x3], axis=-1)
    
    block1_output = Lambda(split_and_extract)(input_layer)
    
    # Transition Convolution
    transition_conv = Conv2D(64, (1, 1), activation='relu')(block1_output)
    
    # Block 2
    block2_output = MaxPooling2D(pool_size=(8, 8))(transition_conv)
    weights = Dense(transition_conv.shape[-1], activation='relu')(Flatten()(block2_output))
    weights = Dense(transition_conv.shape[-1], activation='sigmoid')(weights)
    weights = tf.reshape(weights, (1, 1, 1, transition_conv.shape[-1]))
    main_path_output = tf.multiply(transition_conv, weights)
    
    # Branch to input
    branch_output = input_layer
    
    # Add main path and branch outputs
    combined_output = tf.add(main_path_output, branch_output)
    
    # Final fully connected layer for classification
    final_output = Dense(10, activation='softmax')(combined_output)
    
    # Define and compile the model
    model = Model(inputs=input_layer, outputs=final_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage:
# model = dl_model()
# model.summary()