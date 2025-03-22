import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Dropout, Lambda
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def block(input_tensor):
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1x1)
        dropout = Dropout(0.2)(conv3x3)  # Dropout for feature selection
        return dropout
    
    # Processing each group
    processed_groups = [block(group) for group in split_layer]
    
    # Concatenating the outputs from the three groups
    main_path = Add()(processed_groups)
    
    # Branch pathway: 1x1 convolution to match the output dimension of the main pathway
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Adding the main pathway and branch pathway
    added_output = Add()([main_path, branch_path])
    
    # Flattening the output
    flatten_layer = Flatten()(added_output)
    
    # Fully connected layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model