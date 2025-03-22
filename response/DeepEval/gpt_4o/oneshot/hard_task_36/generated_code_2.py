import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_conv2 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_conv1)
    main_conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_conv2)
    main_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_conv3)
    main_dropout = Dropout(rate=0.5)(main_maxpool)
    
    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_conv)
    
    # Fusion of main and branch pathway
    fused_output = Add()([main_dropout, branch_maxpool])
    
    # Final layers
    global_avg_pool = GlobalAveragePooling2D()(fused_output)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model