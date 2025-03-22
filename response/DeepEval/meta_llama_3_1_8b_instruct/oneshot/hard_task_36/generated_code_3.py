import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, GlobalAveragePooling2D, Add

def dl_model():     

    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    main_pathway_conv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pathway_conv2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway_conv1)
    main_pathway_conv3 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway_conv2)
    main_pathway_maxpool = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(main_pathway_conv3)
    main_pathway_dropout = Dropout(0.5)(main_pathway_maxpool)

    # Branch pathway
    branch_pathway_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu')(main_pathway_maxpool)

    # Fuse the outputs from both pathways
    fused_output = Add()([main_pathway_dropout, branch_pathway_conv])

    # Apply global average pooling
    global_avg_pool = GlobalAveragePooling2D()(fused_output)

    # Apply flattening
    flatten_layer = Flatten()(global_avg_pool)

    # Apply fully connected layer
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model