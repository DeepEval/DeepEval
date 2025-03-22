import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1_1)
    dropout = Dropout(0.5)(max_pooling)
    
    # Branch pathway
    branch_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fusion using addition
    fused = Add()([dropout, branch_conv])
    
    # Global Average Pooling
    gap = GlobalAveragePooling2D()(fused)
    
    # Flatten the output
    flatten_layer = Flatten()(gap)
    
    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model