import keras
from keras.layers import Input, Conv2D, SeparableConv2D, Concatenate, BatchNormalization, Flatten, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def dl_model():     

    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input image into three channel groups
    split_layer = Lambda(lambda x: K.tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Feature extraction using separable convolutional layers
    group1 = SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(split_layer[0])
    group2 = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(split_layer[1])
    group3 = SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(split_layer[2])
    
    # Concatenate the outputs from the three groups
    output_tensor = Concatenate()([group1, group2, group3])
    
    # Batch normalization and flatten
    bath_norm = BatchNormalization()(output_tensor)
    flatten_layer = Flatten()(bath_norm)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model