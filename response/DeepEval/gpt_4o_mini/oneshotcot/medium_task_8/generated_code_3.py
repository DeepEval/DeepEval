import keras
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda, Add, Concatenate
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 color channels
    
    # Main path
    split_input = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    group1 = split_input[0]  # First group remains unchanged
    group2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_input[1])  # Second group with 3x3 conv
    group3 = split_input[2]  # Third group remains unchanged
    
    # Combine group2 and group3
    combined_group = Add()([group2, group3])
    main_path_output = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(combined_group)  # Additional 3x3 conv
    
    # Concatenate all three groups
    main_path_concat = Concatenate()([group1, main_path_output, group3])
    
    # Branch path
    branch_path_output = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)  # 1x1 convolution
    
    # Fuse main and branch paths
    fused_output = Add()([main_path_concat, branch_path_output])
    
    # Final classification
    flatten_layer = Flatten()(fused_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()  # To display the model architecture