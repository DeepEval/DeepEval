import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dropout, Concatenate, Add, Flatten, Dense, Lambda

def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_inputs = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def feature_extraction(input_tensor):
        # Sequence of 1x1 and 3x3 convolutions
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        return conv2
    
    # Process each split input through the feature extraction block
    feature_outputs = [feature_extraction(group) for group in split_inputs]
    
    # Concatenate the outputs to form the main pathway
    main_pathway = Concatenate()(feature_outputs)
    
    # Apply dropout for feature selection
    main_pathway = Dropout(0.5)(main_pathway)

    # Branch pathway
    branch_output = Conv2D(filters=192, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)

    # Combine both pathways using addition
    combined_output = Add()([main_pathway, branch_output])

    # Flatten and fully connected layer for classification
    flatten_layer = Flatten()(combined_output)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model