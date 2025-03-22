import keras
from keras.layers import Input, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D, Dense, Multiply, Flatten, Add
from keras.models import Model

def dl_model():
    # Step 1: Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Step 2: Initial 1x1 convolution to increase channels
    initial_conv = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Depthwise separable convolution
    depthwise_conv = DepthwiseConv2D(kernel_size=(3, 3), padding='same', activation='relu')(initial_conv)
    
    # Step 4: Global Average Pooling
    gap = GlobalAveragePooling2D()(depthwise_conv)
    
    # Step 5: Two fully connected layers for channel attention
    fc1 = Dense(units=depthwise_conv.get_shape().as_list()[2] * 2, activation='relu')(gap)
    fc2 = Dense(units=depthwise_conv.get_shape().as_list()[2], activation='relu')(fc1)
    
    # Step 6: Reshape weights for channel attention weighting
    weights = Dense(units=depthwise_conv.get_shape().as_list()[2], activation='sigmoid')(fc2)
    weights = keras.backend.reshape(weights, (1, 1, depthwise_conv.get_shape().as_list()[2]))
    
    # Step 7: Multiply weights with initial features for channel attention
    weighted_features = Multiply()([depthwise_conv, weights])
    
    # Step 8: 1x1 convolution to reduce dimensionality
    reduced_conv = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu')(weighted_features)
    
    # Step 9: Add the initial input with the weighted features
    output_layer = Add()([reduced_conv, initial_conv])
    
    # Step 10: Flatten layer
    flatten_layer = Flatten()(output_layer)
    
    # Step 11: Fully connected layer for classification
    final_dense = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Build the model
    model = Model(inputs=input_layer, outputs=final_dense)
    
    return model