import keras
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_path = Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_path)
    
    # Branch path
    branch_path = input_layer
    
    # Combine main and branch paths
    output_tensor = Add()([main_path, branch_path])
    
    # Max pooling
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(output_tensor)
    
    flatten_layer = Flatten()(max_pooling)
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=dense_layer)

    return model