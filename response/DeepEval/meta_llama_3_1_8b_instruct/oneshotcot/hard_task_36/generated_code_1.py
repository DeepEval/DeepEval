import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, GlobalAveragePooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2

def dl_model():     
    # Define the input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', 
                    kernel_regularizer=l2(0.01))(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    conv3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv2)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(conv3)
    dropout = Dropout(0.5)(max_pooling)

    # Branch pathway
    conv4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)

    # Fusion pathway
    fusion = Add()([dropout, conv4])
    
    # Global average pooling
    global_avg_pool = GlobalAveragePooling2D()(fusion)
    
    # Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Dense layer
    dense = Dense(units=128, activation='relu')(flatten_layer)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Define the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model