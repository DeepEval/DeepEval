import keras
from keras.layers import Input, Conv2D, Lambda, Dropout, Concatenate, Add, Dense, Flatten
import tensorflow as tf

def dl_model():
    
    def split_channels(x):
        return tf.split(x, num_or_size_splits=3, axis=-1)
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split_layer = Lambda(split_channels)(input_layer)
    
    processed_splits = []
    for i in range(3):
        x = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[i])
        x = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(x)
        x = Dropout(rate=0.3)(x)
        processed_splits.append(x)
    
    # Concatenate the processed groups to form the main pathway
    main_pathway = Concatenate(axis=-1)(processed_splits)
    
    # Branch pathway to match the dimensions of the main pathway
    branch_pathway = Conv2D(filters=96, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch pathways
    combined = Add()([main_pathway, branch_pathway])
    
    # Flatten the result and pass through a fully connected layer
    flatten_layer = Flatten()(combined)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model