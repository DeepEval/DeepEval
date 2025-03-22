import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Branch path
    branch = AveragePooling2D(pool_size=(8, 8), strides=8)(input_layer)
    
    # Block 1
    def block_1(x):
        splits = Lambda(lambda y: tf.split(y, num_or_size_splits=3, axis=3))(x)
        outputs = [Conv2D(filters=x.shape[3]//3, kernel_size=(1, 1), activation='relu')(split) for split in splits]
        return Concatenate()(outputs)
    
    block1_output = block_1(input_layer)
    
    # Block 2
    block2_output = Reshape((block1_output.shape[1], block1_output.shape[2], 3, block1_output.shape[3]//3))(block1_output)
    block2_output = Permute((1, 2, 4, 3))(block2_output)
    block2_output = Reshape(block1_output.shape[1:4].concatenate([3, block1_output.shape[3]//3]))(block2_output)
    
    # Block 3
    block3_output = DepthwiseConv2D(kernel_size=(3, 3), padding='same')(block2_output)
    
    # Block 1 (repeated)
    block1_repeated_output = block_1(block3_output)
    
    # Concatenate main path and branch path
    combined = Concatenate()([block1_repeated_output, branch])
    
    # Flatten and fully connected layers
    flatten_layer = Flatten()(combined)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])