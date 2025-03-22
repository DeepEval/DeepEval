import tensorflow as tf
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # 1x1 convolutions on each group
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split1[0])
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split1[1])
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split1[2])
    
    # Average pooling layers
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv3)
    
    # Concatenate along channel dimension
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten and fully connected layers
    flatten = Flatten()(concat)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model