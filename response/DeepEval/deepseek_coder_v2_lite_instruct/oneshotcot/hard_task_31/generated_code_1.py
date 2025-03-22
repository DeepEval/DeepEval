import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, BatchNormalization, Flatten, Dense, SeparableConv2D, Lambda, Concatenate
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    def first_block(input_tensor):
        # Main Path
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        main_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_path)
        
        # Branch Path
        branch_path = input_tensor
        
        # Addition
        output_tensor = Add()([main_path, branch_path])
        return output_tensor

    first_block_output = first_block(input_layer)
    batch_norm1 = BatchNormalization()(first_block_output)

    # Second Block
    def second_block(input_tensor):
        # Split the input into three groups
        split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        
        # Process each group with different kernel sizes
        outputs = []
        for split in split_layer:
            # 1x1 Convolution
            path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split)
            path1 = Dropout(0.2)(path1)
            
            # 3x3 Convolution
            path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(split)
            path2 = Dropout(0.2)(path2)
            
            # 5x5 Convolution
            path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(split)
            path3 = Dropout(0.2)(path3)
            
            # Concatenate the outputs
            output_tensor = Concatenate()([path1, path2, path3])
            outputs.append(output_tensor)
        
        # Merge the outputs
        final_output = Concatenate()(outputs)
        return final_output

    second_block_output = second_block(batch_norm1)
    batch_norm2 = BatchNormalization()(second_block_output)
    flatten_layer = Flatten()(batch_norm2)
    dense_layer = Dense(units=100, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model