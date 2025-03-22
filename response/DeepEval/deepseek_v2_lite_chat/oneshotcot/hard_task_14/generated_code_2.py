import keras
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, Add, Concatenate

def dl_model():
    # Main Path
    input_layer = Input(shape=(32, 32, 3))  # Assuming 32x32 input image and 3 color channels
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    avg_pool = GlobalAveragePooling2D()(conv1)
    
    # Fully connected layer to generate weights
    fc1 = Dense(units=512, activation='relu')(avg_pool)
    fc2 = Dense(units=256, activation='relu')(fc1)
    
    # Reshape weights to match input layer's shape and multiply element-wise with original feature map
    reshaped_weights = Dense(units=32 * 32 * 3, activation='sigmoid')(fc2)
    reshaped_weights = Dense(shape=(32, 32, 3), activation='tanh')(reshaped_weights)  # Reshape and activate
    weight_map = tf.reshape(reshaped_weights, (-1, 3, 3, 32, 32))  # Reshape weights to match feature map shape
    
    # Multiply element-wise with original feature map
    multiplied_feature_map = tf.multiply(avg_pool, weight_map)
    
    # Branch Path
    branch_input = Conv2D(64, (3, 3), activation='relu')(input_layer)
    branch_output = Add()([avg_pool, branch_input])  # Add main path and branch path outputs
    
    # Fully connected layers
    fc3 = Dense(units=128, activation='relu')(branch_output)
    fc4 = Dense(units=64, activation='relu')(fc3)
    
    # Final output
    output_layer = Dense(units=10, activation='softmax')(fc4)
    
    # Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model