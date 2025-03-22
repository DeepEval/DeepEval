import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    # Step 1: Add input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Step 2: Main pathway
    # 3x3 convolutional layer
    main_path_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # 1x1 convolutional layers
    main_path_conv2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(main_path_conv1)
    main_path_conv3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(main_path_conv2)
    
    # Max pooling
    main_path_pool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path_conv3)
    
    # Dropout layer
    main_path_dropout = Dropout(0.5)(main_path_pool)

    # Step 3: Branch pathway
    # Convolutional layer that matches the size of the output feature maps from the main pathway
    branch_path_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)

    # Step 4: Fusion of main and branch pathways
    fused = Add()([main_path_dropout, branch_path_conv])
    
    # Step 5: Global average pooling layer
    global_avg_pool = GlobalAveragePooling2D()(fused)
    
    # Step 6: Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Step 7: Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Creating the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model