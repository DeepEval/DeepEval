import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model(): 
    
    input_layer = Input(shape=(28, 28, 1))
    
    # Main Pathway
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)
    x = Dropout(0.5)(x) 

    # Branch Pathway
    branch_layer = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x) 

    # Feature Fusion
    x = Add()([x, branch_layer])

    # Global Average Pooling 
    x = GlobalAveragePooling2D()(x)

    # Flatten and Output Layer
    x = Flatten()(x)
    output_layer = Dense(units=10, activation='softmax')(x) 

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model