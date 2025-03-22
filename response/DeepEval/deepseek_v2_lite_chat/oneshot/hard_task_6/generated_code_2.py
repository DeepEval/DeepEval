import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path for the model
    def block1(input_tensor):
        # Block 1: 1x1 convolutions and max pooling
        conv1x1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(input_tensor)
        relu_output = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(conv1x1)
        max_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(relu_output)
        return max_pool
    
    def block2(input_tensor):
        # Block 2: channel shuffling
        reshaped_tensor = Lambda(lambda x: tf.transpose(x, (0, 3, 1, 2)))[0](input_tensor)
        reshaped_tensor = Lambda(lambda x: tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 3)))(reshaped_tensor)
        # Swap channels
        swapped_tensor = Lambda(lambda x: tf.transpose(x, (0, 2, 3, 1)))(reshaped_tensor)
        # Reshape back to original shape
        reshaped_output = Lambda(lambda x: tf.transpose(x, (0, 2, 3, 1)))(swapped_tensor)
        return reshaped_output
    
    def block3(input_tensor):
        # Block 3: depthwise separable convolution
        depthwise = Conv2D(filters=32, kernel_size=(3, 3), padding='same', use_depth_norm=True, depth_multiplier=2)(input_tensor)
        pointwise = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(depthwise)
        return pointwise
    
    # Main path of the model
    main_path_output = block1(input_layer)
    main_path_output = block2(main_path_output)
    main_path_output = block3(main_path_output)
    
    # Branch path: average pooling
    branch_output = MaxPooling2D(pool_size=(4, 4), strides=4, padding='same')(input_layer)
    
    # Concatenate outputs from main path and branch path
    concatenated_output = Concatenate()([main_path_output, branch_output])
    
    # Fully connected layers for classification
    dense1 = Dense(units=256, activation='relu')(concatenated_output)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the deep learning model
model = dl_model()
model.summary()