import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Add, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the channel dimension
    split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Apply 1x1 convolutions to each group independently
    conv1 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_1[0])
    conv2 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_1[1])
    conv3 = Conv2D(filters=input_layer.shape[-1] // 3, kernel_size=(1, 1), padding='same')(split_1[2])
    
    # Downsampling via average pooling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the feature maps along the channel dimension
    concat_layer = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten the concatenated feature maps
    flatten_layer = Flatten()(concat_layer)
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(flatten_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and return the model
model = dl_model()