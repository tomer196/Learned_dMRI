import os

data_path = '/mnt/walkure_pub/Datasets/tomer/h5_1000_new2'
lr = 1e-3
rec_lr = 1e-3
batch_size = 24
init = 'random'
dir_dec = 30
direction_learning = 1
device = '6'
num_epochs = 50

if direction_learning == 1:
    test_name = f'{dir_dec}/{init}_{lr}'
else:
    test_name = f'{dir_dec}/{init}_fixed'

# train
os.system(f'CUDA_VISIBLE_DEVICES={device} python3 train.py --test-name={test_name} --dir-decimation-rate={dir_dec}'
          f' --direction-learning={direction_learning} --sub-lr={lr}  --initialization={init} '
          f'--batch-size={batch_size}  --lr={rec_lr} --num-epochs={num_epochs} '
          f'--data-path={data_path}')

# reconstruct and eval
os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe_nosave.py --test-name={test_name} --data-split=val --batch-size={batch_size}'
          f' --data-path={data_path}')

# reconstruct entire volumes for tractograph algorithm
os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe_vol.py --test-name={test_name} --batch-size={batch_size}'
          f' --data-path={data_path}')
# reconstruct entire volumes of the sub-sample directions using the learned directions for tractograph algorithm
os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe_vol_sub.py --test-name={test_name} --batch-size={batch_size}'
          f' --data-path={data_path}')

# reconstructe and save all the slices
# os.system(f'CUDA_VISIBLE_DEVICES={device} python3 reconstructe.py --test-name={test_name} --data-split=val --batch-size={batch_size}'
#           f'--data-path={data_path}')
# eval all saved slices
# os.system(f'python3 common/evaluate.py --test-name={test_name}'
#           f'--data-path={data_path}')