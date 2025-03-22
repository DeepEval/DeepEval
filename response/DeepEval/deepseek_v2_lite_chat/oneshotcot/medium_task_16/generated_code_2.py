import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv_layers = []
    for i in range(3):
        conv = Conv2D(filters=split_layer[i].shape[-1], kernel_size=(1, 1), padding='same')(split_layer[i])
        conv_layers.append(conv)
    
    # Average pooling for downsampling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_layers[0])
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv_layers[1])
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv_layers[2])
    
    # Concatenate the feature maps along the channel dimension
    concat_layer = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten and pass through two fully connected layers for classification
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model