import daly

def return_dataset(cfg):
    
    annot_data = daly.gen_labels_keyframes(label_tracks=cfg.label_tracks, load_path=cfg.annot_path)

    print('')
    print('Creating training set...')
    
    training_frames = daly.get_frames(annot_data, cfg, split='training') 
    training_set = daly.DALYDataset(annot_data, 
                                     training_frames,
                                     cfg,
                                     split='training')
    
    print('Creating validation set...')
    validation_frames = daly.get_frames(annot_data, cfg, split='validation')
    validation_set = daly.DALYDataset(annot_data, 
                                       validation_frames,
                                       cfg,
                                       split='validation')
    
    print('Finished.')
    print()
    print('%d training clips' % training_set.__len__())
    print('%d validation clips' % validation_set.__len__())
    print()
    
    return training_set, validation_set

