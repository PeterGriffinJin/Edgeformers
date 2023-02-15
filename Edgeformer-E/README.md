# Run the Code
## Train the model

#### Single GPU
```
CUDA_VISIBLE_DEVICES=1 python main.py
```

### Train & Test Edgeformer-E
```
CUDA_VISIBLE_DEVICES=0 python main.py --data_path $data_path --data_mode bert --model_type EdgeformerE --train_batch_size 25 --max_length 256
```

## Test the model

Remember to change args.load_ckpt_name in main.py
```
CUDA_VISIBLE_DEVICES=1 python main.py --data_path Apps/debug --data_mode bert --model_type EdgeformerE --max_length 256 --mode test --load_ckpt_name Apps/debug/ckpt/EdgeformerE-True-1e-05-64-best.pt
```
