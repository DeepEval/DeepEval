import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer in the main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add 1x1 convolutional layer for integrating inter-channel information in the main pathway
    conv1x1_main = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    
    # Step 4: Add max pooling layer in the main pathway
    max_pooling_main = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1x1_main)
    
    # Step 5: Add 50% dropout layer in the main pathway to mitigate overfitting
    dropout_main = Dropout(0.5)(max_pooling_main)
    
    # Step 6: Add convolutional layer in the branch pathway (same size as the output feature maps from the main pathway)
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(dropout_main)
    
    # Step 7: Add addition operation to fuse the outputs from both pathways
    fused = Add()([conv_main, conv_branch])
    
    # Step 8: Add global average pooling layer
    global_avg_pooling = GlobalAveragePooling2D()(fused)
    
    # Step 9: Add flatten layer
    flatten_layer = Flatten()(global_avg_pooling)
    
    # Step 10: Add fully connected layer to produce the classification results
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Step 11: Build the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model