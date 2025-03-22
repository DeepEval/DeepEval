from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    main_conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    main_pool1 = MaxPooling2D(pool_size=(2, 2))(main_conv1)
    main_conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(main_pool1)
    main_pool2 = MaxPooling2D(pool_size=(2, 2))(main_conv2)
    
    # Branch Path
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
    branch_pool = MaxPooling2D(pool_size=(2, 2))(branch_conv)
    
    # Adding the main and branch paths
    added = Add()([main_pool2, branch_pool])
    
    # Flatten and Dense layers
    flatten = Flatten()(added)
    dense1 = Dense(units=128, activation='relu')(flatten)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model