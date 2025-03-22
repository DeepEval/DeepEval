import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, Flatten, Dense
from keras.layers.merge import Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Main path
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_tensor)
        drop1 = Dropout(rate=0.5)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(drop1)
        
        # Branch path
        branch_conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        return Add()([drop1, branch_conv1, pool1])
    
    # Block 2
    def block2(input_tensor):
        # Split input into three groups
        inputs_groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_tensor)
        
        # Process each group with a separable convolution
        conv1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), activation='relu')(inputs_groups[0])
        conv2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), activation='relu')(inputs_groups[1])
        conv3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), activation='relu')(inputs_groups[2])
        
        # Dropout to prevent overfitting
        drop1 = Dropout(rate=0.5)(conv3)
        
        # Concatenate outputs
        concat = Concatenate(axis=-1)([conv1, conv2, drop1, conv3])
        
        # Flatten and fully connected layers
        flatten = Flatten()(concat)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    # Model construction
    model = Model(inputs=input_layer, outputs=[block1(input_tensor=input_layer), block2(input_tensor=input_layer)])
    
    return model

# Instantiate the model
model = dl_model()

# Summary of the model
model.summary()