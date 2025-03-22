import keras
from keras.layers import Input, Conv2D, Dropout, Add, Lambda, SeparableConv2D, Concatenate, Flatten, Dense
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # First Block: Main path with <conv, dropout> and a parallel branch path.
    # Main path
    main_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_dropout1 = Dropout(rate=0.3)(main_conv1)
    main_conv2 = Conv2D(filters=3, kernel_size=(3, 3), padding='same', activation='relu')(main_dropout1)
    
    # Branch path
    branch_path = input_layer
    
    # Adding both paths
    block1_output = Add()([main_conv2, branch_path])
    
    # Second Block: Splitting input and using separable convolutions
    # Splitting along the last dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(block1_output)
    
    # Separable convolutional layers with varying kernel sizes
    conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    dropout1 = Dropout(rate=0.3)(conv1)
    
    conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    dropout2 = Dropout(rate=0.3)(conv2)
    
    conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    dropout3 = Dropout(rate=0.3)(conv3)
    
    # Concatenating the outputs
    block2_output = Concatenate()([dropout1, dropout2, dropout3])
    
    # Final layers
    flatten_layer = Flatten()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model