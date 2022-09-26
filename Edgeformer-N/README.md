# Run the Code
## Train the model

#### Single GPU
```
CUDA_VISIBLE_DEVICES=1 python main.py
```

#### Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 main.py
```

### Train & Test Edgeformer-N
```
CUDA_VISIBLE_DEVICES=3 python main.py --data_path $data_path --data_mode text --model_type EdgeformerN --train_batch_size 25 --max_length 64
```

## Test the model

Remember to change args.load_ckpt_name in main.py
```
python main.py --mode test
```
