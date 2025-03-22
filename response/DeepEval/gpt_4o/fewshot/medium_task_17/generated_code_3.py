import keras
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    def channel_shuffle(input_tensor, groups):
        # Step 1: Reshape the input tensor into (height, width, groups, channels_per_group)
        height, width, channels = input_tensor.shape[1:]
        channels_per_group = channels // groups
        reshaped = tf.reshape(input_tensor, (-1, height, width, groups, channels_per_group))
        
        # Step 2: Swap the third and fourth dimensions
        permuted = tf.transpose(reshaped, perm=[0, 1, 2, 4, 3])
        
        # Step 3: Reshape back to the original shape
        output_tensor = tf.reshape(permuted, (-1, height, width, channels))
        
        return output_tensor
    
    # Apply channel shuffle with 3 groups
    shuffled = Lambda(lambda x: channel_shuffle(x, groups=3))(input_layer)
    
    # Flatten and pass through a fully connected layer
    flatten = Flatten()(shuffled)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()