import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split channels into 3 groups
    channel_splits = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
    
    # Convolutional layers for each group
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(channel_splits[0])
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(channel_splits[1])
    conv3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(channel_splits[2])
    
    # MaxPooling layers for each group
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv1)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv2)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=1, padding='valid')(conv3)
    
    # Concatenate the outputs from all three groups
    fused_features = Concatenate()(list(pool1) + list(pool2) + list(pool3))
    
    # Flatten the fused features
    flattened = Flatten()(fused_features)
    
    # Fully connected layers
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model