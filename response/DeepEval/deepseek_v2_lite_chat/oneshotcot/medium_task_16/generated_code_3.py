import tensorflow as tf
from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, Concatenate

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Split input into three groups along the channel dimension
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(inputs)
    
    # Apply 1x1 convolutions to each group
    conv1 = Conv2D(filters=inputs.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split[0])
    conv2 = Conv2D(filters=inputs.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split[1])
    conv3 = Conv2D(filters=inputs.shape[-1] // 3, kernel_size=(1, 1), padding='same', activation='relu')(split[2])
    
    # Average pooling layers for downsampling
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    # Concatenate the feature maps along the channel dimension
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])
    
    # Flatten the concatenated feature maps
    flat = Flatten()(concat)
    
    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu')(flat)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    return model