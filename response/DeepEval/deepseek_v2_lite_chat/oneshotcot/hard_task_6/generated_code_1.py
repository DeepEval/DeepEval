import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the main path
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split1[0] = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[0])
    split1[1] = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[1])
    split1[2] = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(split1[2])
    fused_features_block1 = keras.backend.concatenate(split1)
    
    # Block 2
    shape = keras.backend.shape(fused_features_block1)
    fused_features_block1 = keras.backend.reshape(fused_features_block1, (shape[0], shape[1]*shape[2], shape[3]))
    fused_features_block1 = keras.backend.permute_dimensions(fused_features_block1, (0, 2, 1))  # Swap third and fourth dimensions
    fused_features_block1 = keras.backend.reshape(fused_features_block1, (shape[0], shape[1], shape[2], shape[3]))
    fused_features_block2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(fused_features_block1)
    
    # Block 3
    fused_features_block3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(fused_features_block2)
    fused_features_block3 = keras.backend.depthwise_conv2d(fused_features_block3, (3, 3), strides=(1, 1), padding='same')
    
    # Main path
    branch_layer = MaxPooling2D(pool_size=(2, 2))(input_layer)
    branch_features = Flatten()(branch_layer)
    
    # Concatenate outputs from main path and branch path
    concatenated_features = Concatenate()([fused_features_block1, fused_features_block2, fused_features_block3, branch_features])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concatenated_features)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create and return the constructed model
model = dl_model()
model.summary()