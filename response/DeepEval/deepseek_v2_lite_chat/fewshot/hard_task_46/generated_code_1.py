import keras
from keras.layers import Input, Lambda, Conv2D, Add, Concatenate, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Splitting the input into three groups and processing each with separable convolutions
    def block1(input_tensor):
        split_1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_1[0])
        conv3x3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_1[1])
        conv5x5_1 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_1[2])
        return Concatenate()([conv1_1, conv3x3_1, conv5x5_1])
    
    # Block 2: Additional branches for enhanced feature extraction
    def block2(input_tensor):
        conv3x3_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_sep_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        depthwise_sep_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        max_pooling_1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='valid')(input_tensor)
        return Concatenate()([conv3x3_2, depthwise_sep_1, depthwise_sep_2, max_pooling_1])
    
    # Processing through both blocks
    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)
    
    # Global average pooling and classification
    avg_pool = GlobalAveragePooling2D()(block2_output)
    output_layer = Dense(units=10, activation='softmax')(avg_pool)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Initialize and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()