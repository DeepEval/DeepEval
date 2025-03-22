import keras
from keras.layers import Input, Conv2D, Dense, Lambda, Add
from keras.layers import Concatenate, Dropout

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_input = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Process each group using a sequence of 1x1 convolution and 3x3 convolution
    group1_output = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[0])
    group1_output = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group1_output)
    group1_output = Dropout(0.2)(group1_output)
    
    group2_output = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[1])
    group2_output = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group2_output)
    group2_output = Dropout(0.2)(group2_output)
    
    group3_output = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_input[2])
    group3_output = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(group3_output)
    group3_output = Dropout(0.2)(group3_output)
    
    # Concatenate the outputs from the three groups to form the main pathway
    main_path_output = Concatenate()([group1_output, group2_output, group3_output])
    
    # Process the input through a 1x1 convolution to match the output dimension of the main pathway
    branch_path_output = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine the outputs from both pathways using an addition operation
    combined_output = Add()([main_path_output, branch_path_output])
    
    # Flatten the combined output and pass it through a fully connected layer for classification
    flatten_output = Flatten()(combined_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_output)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model