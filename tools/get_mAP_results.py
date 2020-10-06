import sys
import os
import torch

results_dir = sys.argv[1]

results = os.listdir(results_dir)
results.remove('config.pkl')
if 'am' in results:
    results.remove('am')
results = sorted(results)
print()
for res in results:
    c = torch.load(os.path.join(results_dir, res), map_location=lambda storage, loc: storage)
    #rint(c['epoch'], round(c['train_loss'], 3), round(c['train_mAP_05'].item(), 3), '|', round(c['val_loss'], 3), round(c['val_mAP_05'], 3))
    print('Epoch', c['epoch'])
    print('-' * 10)
    print('train loss:', round(c['train_loss'], 3), '|', 'train acc.:', round(c['train_acc'].item(), 3), '|', 'train mAP @0.5:', round(c['train_mAP_05'], 3))
    print('val_loss:', round(c['val_loss'], 3), '|', 'val_acc.:', round(c['val_acc'].item(), 3), '|', 'val mAP @0.5:', round(c['val_mAP_05'], 3))
    print()



