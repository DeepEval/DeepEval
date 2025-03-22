import keras
from keras.layers import Input, Lambda, Conv2D, DepthwiseConv2D, MaxPool2D, Add, Concatenate, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split into three channel groups
    channel_groups = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=1))(input_layer)
    
    # Feature extraction for each channel group
    def extract_features(input_tensor):
        # 1x1 separable convolution
        conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        # 3x3 separable convolution
        conv3x3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1x1)
        # 5x5 separable convolution
        conv5x5 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(conv3x3)
        
        return conv1x1, conv3x3, conv5x5
    
    # Extract features for each channel group
    conv1x1, conv3x3, conv5x5 = extract_features(channel_groups[0])
    conv1x1_2, conv3x3_2, conv5x5_2 = extract_features(channel_groups[1])
    conv1x1_3, conv3x3_3, conv5x5_3 = extract_features(channel_groups[2])
    
    # Concatenate feature maps
    concat_layer = Concatenate(axis=-1)([conv1x1, conv3x3, conv5x5, conv1x1_2, conv3x3_2, conv5x5_2, conv1x1_3, conv3x3_3, conv5x5_3])
    
    # Fully connected layers
    dense1 = Dense(units=256, activation='relu')(concat_layer)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()