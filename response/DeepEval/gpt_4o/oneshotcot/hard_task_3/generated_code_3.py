import keras
from keras.layers import Input, Conv2D, Dropout, Lambda, Concatenate, Add, Dense, Flatten
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input along the channel dimension
    split1, split2, split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)

    def feature_extraction_block(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu', padding='same')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(conv1x1)
        dropout = Dropout(rate=0.5)(conv3x3)
        return dropout

    # Process each split group
    group1_output = feature_extraction_block(split1)
    group2_output = feature_extraction_block(split2)
    group3_output = feature_extraction_block(split3)
    
    # Concatenate the outputs of the three groups
    main_pathway = Concatenate()([group1_output, group2_output, group3_output])
    
    # Branch pathway: 1x1 convolution on the input
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), activation='relu', padding='same')(input_layer)

    # Combine the main and branch pathways
    combined_output = Add()([main_pathway, branch_pathway])
    
    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model