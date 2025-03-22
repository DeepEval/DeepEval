import keras
from keras.layers import Input, Conv2D, Lambda, Concatenate, BatchNormalization, Flatten, Dense, tf

def dl_model():  
    
    input_layer = Input(shape=(32, 32, 3)) 
    
    # Main Path
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)  
    
    # Feature Extraction
    path1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(x[0])
    path1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(path1)
    path1 = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(path1) 
    
    path2 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(x[1])
    path2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(path2)
    path2 = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(path2) 

    path3 = Conv2D(filters=16, kernel_size=(1, 1), padding='same', activation='relu')(x[2])
    path3 = Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu')(path3)
    path3 = Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu')(path3)

    # Concatenate outputs
    main_path_output = Concatenate(axis=2)([path1, path2, path3])

    # Branch Path
    branch_path = Conv2D(filters=48, kernel_size=(1, 1), padding='same', activation='relu')(input_layer) 

    # Fusion
    fused_output = main_path_output + branch_path

    # Flatten and Fully Connected Layers
    flatten_layer = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model