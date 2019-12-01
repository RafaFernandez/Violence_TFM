

def train_base_model(learning_rate, epcs, batch_size, db_dir, exper_dir, gpu_rate, weights_path='imagenet',
                     n_dense_layer=1, dense_neurons=256, drop_value=0.2,include_top=True):
    """
    Main method to train base model

    :param learning_rate: Learning rate use for training
    :param epcs: Number of epochs to train
    :param batch_size: Batch size to be loaded
    :param db_dir: Path to the dataset
    :param exper_dir: Path to directory where results will be store
    :param gpu_rate: GPU memory usage rate
    :param weights_path: Path to pretrained model weights
    :param n_dense_layer: Number of dense layer
    :param dense_neurons: Number of neurons in dense layer
    :param drop_value: Dropout value
    :param include_top: Whether or not include top layers
    """

    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    tf.set_random_seed(1330)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_rate
    config.gpu_options.visible_device_list = "0"
    set_session(tf.Session(config=config))

    from keras import callbacks
    from Violence.model_callbacks import metric_callback
    from Violence.model_net import generate_model_VGG16
    from Violence.model_train import train_model, create_generator
    from keras.applications.vgg16 import preprocess_input
    import numpy as np

    #for reproducibility
    np.random.seed(1337)
    img_width, img_height = 224, 224

    #Get data generator for database
    model=generate_model_VGG16(weights_path=weights_path,
                               input_shape=(img_width,img_height,3),
                               n_dense_layer=n_dense_layer,
                               drop_value=drop_value,
                               dense_neurons=dense_neurons,
                               out_neurons=4,
                               activation='softmax',
                               fine_tuning=True,
                               include_top=include_top)

    train_dir=db_dir+"/train/"
    val_dir=db_dir+"/validation/"
    out_dir=exper_dir+"/results"
    train_generator = create_generator(data_dir=train_dir,target_size=(img_width,img_height), batch_size=batch_size, color_mode='rgb', preprocess_input=preprocess_input,shuffle=True)
    val_generator = create_generator(data_dir=val_dir, batch_size=batch_size, target_size=(img_width,img_height), color_mode='rgb', preprocess_input=preprocess_input,shuffle=False)
    #Callbacks
    log_file = out_dir+"/training_lr"+str(learning_rate)+"_dL"+str(n_dense_layer)+"_dN"+str(dense_neurons)+"_dr"+str(drop_value)+".csv"
    matrix_file =  out_dir+"/matrix_lr"+str(learning_rate)+"_dL"+str(n_dense_layer)+"_dN"+str(dense_neurons)+"_dr"+str(drop_value)+".txt"
    early_stopping = callbacks.EarlyStopping(monitor='val_loss',min_delta=0, patience=100, verbose=0, mode='auto')
    save_log_file = callbacks.CSVLogger(log_file,separator=" ")
    metrics = metric_callback(validation_generator=val_generator,filename=out_dir+"/models/model"+str(learning_rate)+"_dL"+str(n_dense_layer)+"_dN"+str(dense_neurons)+"_dr"+str(drop_value)+".hdf5",filename_matrix=matrix_file)

    callbacks = [early_stopping, metrics, save_log_file]

    # Fit the model
    train_model(model,
                train_samples=train_generator.samples,
                train_generator=train_generator,
                val_generator=val_generator,
                val_samples=val_generator.samples,
                callbacks=callbacks,
                epochs=epcs,
                batch_size=batch_size,
                lr=learning_rate)

    return model


if __name__ == '__main__':
    train_base_model(learning_rate=0.001, drop_value=0, n_dense_layer=1, batch_size=1, dense_neurons=128, epcs=8,
             weights_path='imagenet',
             db_dir='',
             exper_dir='', gpu_rate=1.0,include_top=True)
