# Gene Inference

1、under \model are many different network models.  
2、under \pytorch is our training code.  
3、under \utils are some tool files.  

you can run this command to run train.py  
```
CUDA_VISIBLE_DEVICES=0 python train.py --model 'relu_3' --num-epoch 200 --batch-size 5000 --in-size 943 --out-size 4760 --hidden-size 1000 --dropout-rate 0.1 --learning-rate 5e-4 --dataset '0-4760'
```
