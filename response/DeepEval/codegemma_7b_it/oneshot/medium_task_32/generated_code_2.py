import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense
from keras.initializers import he_normal

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Feature extraction for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=he_normal())(split_layer[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer=he_normal())(split_layer[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer=he_normal())(split_layer[2])
    
    # Concatenate the outputs of the three groups
    concat_layer = Concatenate()([conv1, conv2, conv3])
    
    # Fuse the features
    fusion_layer = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_initializer=he_normal())(concat_layer)
    
    # Flatten and classify
    flatten_layer = Flatten()(fusion_layer)
    dense_layer = Dense(units=10, activation='softmax', kernel_initializer=he_normal())(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=dense_layer)
    
    return model