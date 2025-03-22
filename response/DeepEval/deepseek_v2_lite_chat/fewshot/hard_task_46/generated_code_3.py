import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Flatten, Concatenate, AveragePooling2D, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for the first block
    split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # First block: three separate paths
    def block1(input_tensor):
        # 1x1 separable convolution
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor[0])
        # 3x3 separable convolution
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor[1])
        # 5x5 separable convolution
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(input_tensor[2])
        
        # Concatenate the outputs
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat1
    
    block1_output = block1(input_tensor=split1)
    
    # Second block
    def block2(input_tensor):
        # 3x3 convolution
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # Branch 1: 1x1 -> 3x3 -> 3x3
        conv2_branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        conv2_branch1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2_branch1)
        # Branch 2: 1x1 -> 1x1 -> MaxPool2D
        conv2_branch2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv2_branch2)
        # Concatenate the outputs
        concat2 = Concatenate(axis=-1, name='concat_branch2')([conv2_branch1, pool2])
        
        # Flatten and fully connected layers
        flat = Flatten()(concat2)
        dense = Dense(units=128, activation='relu')(flat)
        output = Dense(units=10, activation='softmax')(dense)
        
        return output
    
    # Model output
    model_output = block2(block1_output)
    
    # Return the model
    model = keras.Model(inputs=input_layer, outputs=model_output)
    return model

# Create the model
model = dl_model()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()