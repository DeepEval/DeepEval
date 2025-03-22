import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Main Pathway
    # 3x3 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Two 1x1 convolutional layers
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
    
    # Max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    
    # Dropout layer
    dropout = Dropout(rate=0.5)(max_pooling)

    # Step 3: Branch Pathway
    # Convolutional layer that matches the size of the output feature maps from the main pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Step 4: Fusion of both pathways
    # Addition of the outputs from both pathways
    fused_output = Add()([dropout, branch_conv])
    
    # Step 5: Global Average Pooling layer
    global_avg_pooling = GlobalAveragePooling2D()(fused_output)
    
    # Step 6: Flatten layer
    flatten_layer = Flatten()(global_avg_pooling)
    
    # Step 7: Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Build the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model