import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Add
from keras.models import Model
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_path = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(input_layer)
    conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_path[0])
    conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path[1])
    conv5x5 = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path[2])
    main_output = Concatenate()([conv1x1, conv3x3, conv5x5])
    
    # Branch Path
    branch_conv = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    branch_output = Add()([main_output, branch_conv])
    
    # Batch Normalization and Flatten
    batch_norm = BatchNormalization()(branch_output)
    flatten_layer = Flatten()(batch_norm)
    
    # Fully Connected Layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model