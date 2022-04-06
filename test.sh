
docker build -t torchvision .

docker run -p 8000:8000 torchvision

docker build -t torchvision .
docker tag torchvision:latest 681747700094.dkr.ecr.ap-northeast-2.amazonaws.com/torchvision:latest
aws ecr get-login-password --region ap-northeast-2 | docker login --username AWS --password-stdin 681747700094.dkr.ecr.ap-northeast-2.amazonaws.com
docker push 681747700094.dkr.ecr.ap-northeast-2.amazonaws.com/torchvision:latest



vision-api.

curl -X POST -F file=@cat_pic.jpeg https://vision-api.octankshop.com/predict