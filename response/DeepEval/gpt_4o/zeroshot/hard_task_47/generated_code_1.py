import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Splitting input and using depthwise separable convolutions
    def split_input(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    split_layers = Lambda(split_input)(input_layer)
    
    conv_layers = []
    for size, layer in zip([1, 3, 5], split_layers):
        conv = SeparableConv2D(32, (size, size), padding='same', activation='relu')(layer)
        conv = BatchNormalization()(conv)
        conv_layers.append(conv)
    
    concat_first_block = Concatenate()(conv_layers)
    
    # Second Block: Multiple branches
    branch1 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat_first_block)
    branch1 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch1)

    branch2 = Conv2D(64, (1, 1), activation='relu', padding='same')(concat_first_block)
    branch2 = Conv2D(64, (1, 7), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (7, 1), activation='relu', padding='same')(branch2)
    branch2 = Conv2D(64, (3, 3), activation='relu', padding='same')(branch2)

    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(concat_first_block)
    
    concat_second_block = Concatenate()([branch1, branch2, branch3])
    
    # Fully connected layers for classification
    flat = Flatten()(concat_second_block)
    dense1 = Dense(256, activation='relu')(flat)
    output_layer = Dense(10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()

# Print the model summary
model.summary()