import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, BatchNormalization, Flatten, Dense, Reshape
from keras.regularizers import l2

def dl_model():     
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Main path: Global Average Pooling and Weight Generation
    main_output = AveragePooling2D(pool_size=(8, 8), strides=1, padding='valid')(input_layer)
    main_output = Reshape((3,))(main_output)
    weight1 = Dense(units=64, activation='relu', kernel_regularizer=l2(0.01))(main_output)
    weight1 = Dense(units=64, activation='relu', kernel_regularizer=l2(0.01))(weight1)
    weight1 = Reshape((64,))(weight1)
    
    # Element-wise multiplication of weights with the original feature map
    element_wise_product = keras.layers.Multiply()([input_layer, weight1])
    
    # Branch path: Convolutional Path
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(input_layer)
    branch_output = Conv2D(filters=3, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(branch_output)

    # Add outputs from main and branch paths
    combined_output = keras.layers.Add()([element_wise_product, branch_output])
    
    # Apply Batch Normalization and Flatten
    combined_output = BatchNormalization()(combined_output)
    combined_output = Flatten()(combined_output)

    # Output Layers
    dense1 = Dense(units=128, activation='relu')(combined_output)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model