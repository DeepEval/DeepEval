import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Add, GlobalAveragePooling2D, Flatten, Dense

def dl_model():
    
    # Step 1: Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    main_path = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_path)
    main_path = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_path)
    main_path = Dropout(rate=0.5)(main_path)
    
    # Branch pathway
    branch_path = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Adding both pathways
    combined = Add()([main_path, branch_path])
    
    # Global Average Pooling layer
    global_avg_pool = GlobalAveragePooling2D()(combined)
    
    # Flatten layer
    flatten_layer = Flatten()(global_avg_pool)
    
    # Fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model