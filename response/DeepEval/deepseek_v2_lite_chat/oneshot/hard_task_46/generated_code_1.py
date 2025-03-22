import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel axis
    channel_splits = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    
    # First block
    for i, split in enumerate(channel_splits):
        conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split)
        conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), strides=1, padding='same')(conv3)
        
    # Concatenate outputs from each branch
    concat = Concatenate()(channel_splits)
    
    # Second block
    conv4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    conv5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
    conv6 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv5)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv6)
    
    # Global average pooling
    avg_pool = GlobalAveragePooling2D()(pool2)
    
    # Fully connected layer
    dense = Dense(units=512, activation='relu')(avg_pool)
    outputs = Dense(units=10, activation='softmax')(dense)
    
    # Model
    model = Model(inputs=input_layer, outputs=outputs)
    
    return model