import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Separate convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_groups[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_groups[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_groups[2])
    
    # Max pooling
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv3)
    
    # Concatenate
    concat = Concatenate()(
        [tf.expand_dims(pool1, axis=2), tf.expand_dims(pool2, axis=2), tf.expand_dims(pool3, axis=2)])
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

model = dl_model()
model.summary()