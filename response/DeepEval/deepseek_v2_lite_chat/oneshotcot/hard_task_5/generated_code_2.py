import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Permute
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Block 1: Split into three groups and process each with a 1x1 convolutional layer
    split = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    conv1 = [Conv2D(filters=int(32/3), kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', name='conv1_' + str(i+1))(group) for i, group in enumerate(split)]
    
    # Concatenate the outputs along the channel dimension
    fused_features = Concatenate(axis=-1)(conv1)
    
    # Block 2: Channel shuffle
    reshape = Lambda(lambda x: tf.transpose(x, perm=[0, 3, 1, 2]))(fused_features)
    reshape_fused_features = reshape(fused_features)
    channel_shuffle = Permute((2, 3, 1), name='permute_block2')(reshape_fused_features)
    
    # Block 3: Depthwise separable convolution
    depthwise_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', depthwise_initializer='zeros', activation='relu')(channel_shuffle)
    
    # Branch directly connected to the input
    branch_output = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    
    # Combine outputs from the main path and the branch
    combined_features = keras.layers.Add()([depthwise_conv, branch_output])
    
    # Final dense layers for classification
    flat = Flatten()(combined_features)
    dense1 = Dense(units=128, activation='relu')(flat)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])