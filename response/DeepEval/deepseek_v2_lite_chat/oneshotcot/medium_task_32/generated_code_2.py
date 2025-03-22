import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Split the input into three groups along the last dimension
    group1, group2, group3 = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Feature extraction with depthwise separable convolutional layers
    conv1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', activation='relu')(group1)
    conv2 = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(group2)
    conv3 = Conv2D(filters=32, kernel_size=5, strides=1, padding='same', activation='relu')(group3)
    
    # Concatenate and merge the feature maps
    fused_features = Concatenate(axis=-1)([conv1, conv2, conv3])
    
    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(fused_features)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model