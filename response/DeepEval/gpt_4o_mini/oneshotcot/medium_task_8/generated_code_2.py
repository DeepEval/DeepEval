import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense, Add
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB

    # Main path
    split_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # First group remains unchanged
    group1 = split_groups[0]
    
    # Second group with 3x3 convolution
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_groups[1])
    
    # Combine second group with third group
    combined_group = Add()([group2, split_groups[2]])  # Element-wise addition
    main_path_output = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(combined_group)
    
    # Concatenate all three groups
    main_path_final = Concatenate()([group1, main_path_output, split_groups[2]])

    # Branch path
    branch_path_output = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main path and branch path
    combined_output = Add()([main_path_final, branch_path_output])

    # Flatten and fully connected layer
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # 10 classes for CIFAR-10

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model