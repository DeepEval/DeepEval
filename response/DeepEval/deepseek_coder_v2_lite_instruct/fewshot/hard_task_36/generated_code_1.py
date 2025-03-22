import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    conv_main = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv_int_main = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv_main)
    pool_main = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv_int_main)
    dropout_main = Dropout(0.5)(pool_main)
    
    # Branch pathway
    conv_branch = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fusion of both pathways
    added = Add()([dropout_main, conv_branch])
    
    # Final processing
    global_avg_pool = GlobalAveragePooling2D()(added)
    flatten_layer = Flatten()(global_avg_pool)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model