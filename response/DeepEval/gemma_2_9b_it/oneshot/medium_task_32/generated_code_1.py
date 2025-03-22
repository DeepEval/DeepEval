import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3)) 

    # Split the input into three groups
    split_tensor = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer) 

    # Feature extraction with depthwise separable convolutions
    group1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
    group2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
    group3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])

    # Concatenate the outputs of the three groups
    fused_features = Concatenate()([group1, group2, group3])

    # Flatten and classification
    flatten_layer = Flatten()(fused_features)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model