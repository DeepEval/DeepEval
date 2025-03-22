import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Lambda, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Dual-path convolutional network
    def block1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(pool1)
        branch_pool = MaxPooling2D(pool_size=(2, 2))(input_tensor)
        
        # Branch path
        branch_conv = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(branch_pool)
        
        # Add main path and branch path
        add_layer = Add()([conv2, branch_conv])
        
        # Flatten and fully connected layers
        flatten = Flatten()(add_layer)
        dense = Dense(units=512, activation='relu')(flatten)
        
        return dense
    
    # Block 2: Feature extraction using depthwise separable convolutions
    def block2(input_tensor):
        # Split input into three groups
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Depthwise separable convolutions
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), activation=keras.activations.relu)(split1[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), activation=keras.activations.relu)(split2[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), activation=keras.activations.relu)(split3[2])
        
        # Concatenate feature maps
        concat = Concatenate(axis=-1)([conv1, conv2, conv3])
        
        # Fully connected layers
        flatten = Flatten()(concat)
        dense1 = Dense(units=256, activation='relu')(flatten)
        dense2 = Dense(units=128, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=block2(block1(input_tensor=input_layer)))
    
    return model

# Create the model
model = dl_model()

# Display model summary
model.summary()