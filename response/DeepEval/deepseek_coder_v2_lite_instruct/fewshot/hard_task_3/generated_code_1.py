import keras
from keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense, Dropout
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Splitting the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    def pathway(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(conv1)
        dropout = Dropout(0.5)(conv2)
        return dropout
    
    # Main pathway for each group
    main_pathway1 = pathway(split_layer[0])
    main_pathway2 = pathway(split_layer[1])
    main_pathway3 = pathway(split_layer[2])
    
    # Branch pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # Addition operation
    added = Add()([main_pathway1, main_pathway2, main_pathway3, branch_pathway])
    
    # Flatten and fully connected layer
    flatten_layer = Flatten()(added)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model