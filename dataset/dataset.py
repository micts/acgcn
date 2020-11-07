from dataset import daly

def return_dataset(data_path, annot_path, cfg, split):
    
    print('Preparing {} set...'.format(split))
    annot_data = daly.load_tracks(load_path=annot_path)
    frames = daly.get_frames(annot_path, annot_data, cfg, split=split, on_keyframes=False) 
    dataset = daly.DALYDataset(data_path,
                               annot_data, 
                               frames,
                               cfg,
                               split=split)
    print('Finished.\n')    


    #print('Creating validation set...')
    #validation_frames = daly.get_frames(cfg.annot_path, annot_data, cfg, split='validation')
    #validation_set = daly.DALYDataset(cfg.data_path,
#				      annot_data, 
#                                      validation_frames,
#                                      cfg,
#                                      split='validation')
    
    #print('Finished.') 
    return dataset

