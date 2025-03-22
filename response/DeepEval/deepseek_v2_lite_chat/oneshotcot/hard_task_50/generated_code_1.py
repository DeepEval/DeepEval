import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Concatenate, BatchNormalization, SeparableConv2D, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    def block1(input_tensor):
        # Three max pooling layers with different scales
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=1, padding='valid')(input_tensor)
        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_tensor)
        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=4, padding='valid')(input_tensor)
        
        # Flatten and dropout for regularization
        flat1 = Flatten()(maxpool3)
        drop1 = keras.layers.Dropout(rate=0.5)(flat1)
        
        # Reshape to 4D tensor
        reshape1 = keras.layers.Reshape((-1, drop1.shape[-1]))(drop1)
        
        return reshape1
    
    block1_output = block1(input_layer)
    
    # Block 2
    def block2(input_tensor):
        # Split the input into four groups
        split1 = Lambda(lambda x: keras.backend.split(x, 4, axis=-1))(input_tensor)
        
        # Process each group with a separable convolution
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(split1[0])
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(split1[1])
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(split1[2])
        conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), padding='same')(split1[3])
        
        # Concatenate the outputs from the separable convolutions
        concat = Concatenate()(conv1, conv2, conv3, conv4)
        
        # Flatten and dense layers
        flatten = Flatten()(concat)
        dense1 = Dense(units=128, activation='relu')(flatten)
        dense2 = Dense(units=64, activation='relu')(dense1)
        output_layer = Dense(units=10, activation='softmax')(dense2)
        
        return output_layer
    
    model = Model(inputs=input_layer, outputs=block2(block1_output))
    
    return model

# Construct the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
model.summary()