import keras
from keras.layers import Input, Lambda, Conv2D, MaxPool2D, Flatten, Dense, Add
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split_layer = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group and downsample
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_layer[0])
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_layer[1])
    pool2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_layer[2])
    pool3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)
    
    # Concatenate feature maps along the channel dimension
    concat_layer = Add()([pool1, pool2, pool3])
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(concat_layer)
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Return the constructed model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Instantiate and return the model
model = dl_model()