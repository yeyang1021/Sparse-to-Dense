
This is a repo [sparse to dense (Self-supervised Sparse-to-Dense: Self-supervised Depth Completion from LiDAR and Monocular Camera)] modified by [lanenet-lane-detection](https://github.com/MaybeShewill-CV/lanenet-lane-detection)

The trained model achieves similar results in KITTI dataset


You can test a single image on the trained model as follows

```
python tools/my1test_debse.py --is_batch False --batch_size 1 
--weights_path model/***.ckpt-94000 
--image_path data/***/0.jpg
```


#### Train model
You may call the following script to train your own model

```
python tools/my1train_dense.py --net resnet --dataset_dir data/training_data_example/
```
You can also continue the training process from the snapshot by
```
python tools/my1train_dense.py --net resnet --dataset_dir /path/to/your/data --weights_path path/to/your/last/checkpoint
```


