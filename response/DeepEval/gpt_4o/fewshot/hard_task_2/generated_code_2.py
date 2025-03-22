import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Add, Flatten, Dense, Lambda

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def group_conv_block(group_input):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(group_input)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
        return conv3

    # Process each group through the convolutional block
    group1_output = group_conv_block(groups[0])
    group2_output = group_conv_block(groups[1])
    group3_output = group_conv_block(groups[2])
    
    # Combine the group outputs using addition to form the main path
    main_path = Add()([group1_output, group2_output, group3_output])
    
    # Fuse the main path with the original input
    fused_output = Add()([main_path, input_layer])

    # Flatten and then pass through a fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model