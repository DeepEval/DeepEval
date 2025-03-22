import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D

def dl_model():
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Add convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # Step 3: Add max pooling layer
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    
    # Step 4: Define a block for the main pathway
    def block(input_tensor):
        # Step 4.1: Add 1x1 convolutional layer
        conv2 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(input_tensor)
        # Step 4.2: Add another 1x1 convolutional layer
        conv3 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu')(conv2)
        # Step 4.3: Add max pooling layer
        max_pooling = MaxPooling2D(pool_size=(1, 1), strides=1, padding='same')(input_tensor)
        # Step 4.4: Concatenate the outputs
        concat_tensor = Concatenate(axis=-1)([conv2, conv3, max_pooling])
        return concat_tensor
    
    main_pathway_output = block(max_pooling)
    batch_norm = BatchNormalization()(main_pathway_output)
    dropout = Dropout(0.5)(batch_norm)  # 50% dropout
    
    # Flatten and dense layers
    flatten_layer = Flatten()(dropout)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Branch pathway
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_output = block(branch_conv)
    
    # Fusion of main and branch pathways
    fused_tensor = Concatenate(axis=-1)([main_pathway_output, branch_output])
    
    # Global average pooling and final dense layer
    avg_pool = AveragePooling2D(pool_size=(7, 7))(fused_tensor)
    flatten = Flatten()(avg_pool)
    output = Dense(units=10, activation='softmax')(flatten)
    
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model