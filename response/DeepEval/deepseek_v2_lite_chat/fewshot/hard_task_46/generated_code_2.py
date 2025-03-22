import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, SeparableConv2D, MaxPool2D, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # First block
    def block1(input_tensor):
        # Split input into three groups
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with different convolutional layers
        conv1_1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split1[0])
        conv1_2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(split1[1])
        conv1_3 = SeparableConv2D(filters=64, kernel_size=(5, 5), activation='relu')(split1[2])
        
        # Concatenate the outputs
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat1
    
    # Second block
    def block2(input_tensor):
        # Multiple branches for feature extraction
        conv2_1 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2_2 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu)(input_tensor)
        conv2_3 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu')(input_tensor)
        conv2_4 = MaxPool2D(pool_size=(2, 2))(input_tensor)
        
        # Concatenate the outputs
        concat2 = Concatenate(axis=-1)([conv2_1, conv2_2, conv2_3, conv2_4])
        return concat2
    
    # Process through both blocks and output layer
    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)
    flat = Flatten()(block2_output)
    dense = Dense(units=256, activation='relu')(flat)
    output = Dense(units=10, activation='softmax')(dense)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Instantiate and return the model
model = dl_model()