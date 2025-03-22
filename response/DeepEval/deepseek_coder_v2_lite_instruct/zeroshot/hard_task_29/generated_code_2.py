import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Add

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # First block
    # Main path
    main_path = Conv2D(32, (3, 3), activation='relu')(input_layer)
    main_path = Conv2D(32, (3, 3), activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine outputs from both paths
    combined = Add()([main_path, branch_path])
    
    # Second block
    # Pooling layers
    pool1 = MaxPooling2D(pool_size=(1, 1))(combined)
    pool2 = MaxPooling2D(pool_size=(2, 2))(combined)
    pool3 = MaxPooling2D(pool_size=(4, 4))(combined)
    
    # Flatten the results
    flatten1 = Flatten()(pool1)
    flatten2 = Flatten()(pool2)
    flatten3 = Flatten()(pool3)
    
    # Concatenate the flattened results
    concatenated = tf.concat([flatten1, flatten2, flatten3], axis=-1)
    
    # Fully connected layers
    dense1 = Dense(128, activation='relu')(concatenated)
    dense2 = Dense(10, activation='softmax')(dense1)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=dense2)
    
    return model

# Example usage
model = dl_model()
model.summary()