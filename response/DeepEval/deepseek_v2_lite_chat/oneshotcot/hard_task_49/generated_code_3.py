import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Block 1
    def block1(input_tensor):
        # Average pooling layers with different scales
        avg_pool_1x1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_tensor)
        avg_pool_2x2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_tensor)
        avg_pool_4x4 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_tensor)
        # Flatten and concatenate
        flat1 = Flatten()(avg_pool_1x1)
        flat2 = Flatten()(avg_pool_2x2)
        flat3 = Flatten()(avg_pool_4x4)
        concat = Concatenate()([flat1, flat2, flat3])
        
        # Transformation to 4D tensor for second block
        reshape = Flatten()(concat)
        reshape_tensor = Lambda(lambda x: x.reshape((-1, 16)))(reshape)
        
        return reshape_tensor
    
    # Block 2
    def block2(input_tensor):
        # Split input into four groups
        split1 = Lambda(lambda x: keras.layers.split(x, 4, axis=-1))(input_tensor)
        
        # Depthwise separable convolutional layers
        conv1_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_5x5 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        conv1_7x7 = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split1[3])
        # Concatenate
        concat1 = Concatenate(axis=-1)([conv1_1x1, conv1_3x3, conv1_5x5, conv1_7x7])
        
        # Similar processing for other groups
        conv2_1x1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv2_3x3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv2_5x5 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        conv2_7x7 = Conv2D(filters=128, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(split1[3])
        concat2 = Concatenate(axis=-1)([conv2_1x1, conv2_3x3, conv2_5x5, conv2_7x7])
        
        # Concatenate the results
        concat23 = Concatenate(axis=-1)([concat1, concat2])
        
        # Flatten and dense layers
        flat_concat = Flatten()(concat23)
        dense = Dense(units=128, activation='relu')(flat_concat)
        output = Dense(units=10, activation='softmax')(dense)
        
        model = Model(inputs=input_layer, outputs=output)
        return model
    
    # Model construction
    model = block1(input_layer)
    model = block2(model)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])