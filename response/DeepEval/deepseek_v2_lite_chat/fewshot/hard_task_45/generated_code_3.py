import keras
from keras.layers import Input, Lambda, Concatenate, Conv2D, DepthwiseConv2D, MaxPooling2D, Flatten, Dense, Reshape

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Splitting the input into three groups and applying depthwise separable convolutions
    def block1(input_tensor):
        split1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = DepthwiseConv2D(kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = DepthwiseConv2D(kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        concat1 = Concatenate(axis=-1)([conv1_1, conv1_2, conv1_3])
        return concat1
    
    # Block 2: Multiple branches for feature extraction
    def block2(input_tensor):
        conv2_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        conv2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        conv2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(maxpool)
        concat2 = Concatenate(axis=-1)([conv2_1, conv2_2, conv2_3])
        return concat2
    
    # Construct the model
    block1_output = block1(input_tensor=input_layer)
    block2_output = block2(input_tensor=block1_output)
    
    # Flatten and fully connected layers
    flatten = Flatten()(block2_output)
    dense = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])