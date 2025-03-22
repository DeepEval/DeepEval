from keras.layers import Input, Conv2D, MaxPooling2D, Add, Dropout, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Input layer
    input_layer = Input(shape=(28, 28, 1))
    
    # Main pathway
    main_pathway = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    main_pathway = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_pathway)
    main_pathway = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2), padding='same')(main_pathway)
    main_pathway = Dropout(0.5)(main_pathway)
    
    # Branch pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    branch_pathway = MaxPooling2D(pool_size=(2, 2), padding='same')(branch_pathway)
    
    # Fuse both pathways using an addition operation
    fused_output = Add()([main_pathway, branch_pathway])
    
    # Final layers
    global_avg_pooling = GlobalAveragePooling2D()(fused_output)
    flatten_layer = Flatten()(global_avg_pooling)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model