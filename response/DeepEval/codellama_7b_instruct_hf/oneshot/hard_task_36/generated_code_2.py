import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

def dl_model():
    # Main pathway
    main_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    main_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway)
    main_pathway = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_pathway)
    main_pathway = Dropout(rate=0.5)(main_pathway)
    # Branch pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(input_layer)
    branch_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pathway)
    branch_pathway = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch_pathway)
    branch_pathway = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_pathway)
    # Fuse the two pathways
    fused_pathway = keras.layers.concatenate([main_pathway, branch_pathway])
    # Additional layers
    fused_pathway = Flatten()(fused_pathway)
    fused_pathway = Dense(units=128, activation='relu')(fused_pathway)
    fused_pathway = Dense(units=64, activation='relu')(fused_pathway)
    output_layer = Dense(units=10, activation='softmax')(fused_pathway)
    # Create the model
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    return model