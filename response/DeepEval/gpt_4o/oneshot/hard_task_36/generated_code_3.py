import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Pathway
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_conv2)
    main_max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(main_conv3)
    main_dropout = Dropout(rate=0.5)(main_max_pooling)
    
    # Branch Pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_layer)
    
    # Fusion via addition
    fused = Add()([main_dropout, branch_conv])
    
    # Global Average Pooling and Flatten
    global_avg_pooling = GlobalAveragePooling2D()(fused)
    flatten_layer = Flatten()(global_avg_pooling)
    
    # Fully Connected Layer for classification
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Model creation
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model