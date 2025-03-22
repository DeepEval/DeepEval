import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, BatchNormalization, Flatten, Dense, Lambda, SeparableConv2D
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block
    def block1(input_tensor):
        # Main path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        
        # Branch path
        branch_path = input_tensor
        
        # Add the outputs of both paths
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    block1_output = block1(input_layer)

    # Second block
    def block2(input_tensor):
        # Split the input into three groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with separable convolutional layers
        paths = []
        for i, group in enumerate(split_layer):
            if i == 0:
                path = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group)
            elif i == 1:
                path = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group)
            elif i == 2:
                path = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(group)
            
            path = Dropout(0.5)(path)
            paths.append(path)
        
        # Concatenate the outputs of the three groups
        output_tensor = Add()(paths)
        return output_tensor

    block2_output = block2(block1_output)

    # Flatten and fully connected layer
    flatten_layer = Flatten()(block2_output)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model