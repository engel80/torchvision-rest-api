# Setup on GPU EC2 instance
```bash
cd ~
git clone https://github.com/engel80/torchvision-rest-api
cd torchvision-rest-api
git checkout init-commit
git pull
chmod +x *.sh
chmod +x *.py
pip3 install -r requirements.txt

alias smi="watch -d -n 1 nvidia-smi"
alias gpustat="watch -d -n 1 /home/ssm-user/.local/bin/gpustat"
```

# Running

```bash
python app.py
```

# Test

From another tab, run the smi for "watch -d -n 1 nvidia-smi"

```bash
smi
```

```bash
curl http://127.0.0.1:9000/gputest
```

```bash
curl -X POST -F file=@cat_pic.jpeg http://127.0.0.1:9000/predict
```

# Test with gpu_burn image

```bash
cd ~
git clone https://github.com/wilicc/gpu-burn
cd gpu-burn
sudo docker build -t gpu_burn .
sudo docker run --rm --gpus all gpu_burn
```

## Refeerence

https://github.com/avinassh/pytorch-flask-api