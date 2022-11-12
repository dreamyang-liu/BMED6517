backbone=resnet18

resize=256
lr=0.001
epochs=10
batch_size=64

python main.py \
--backbone ${backbone} \
--lr ${lr} \
--epochs ${epochs} \
--batch_size ${batch_size} \
--resize ${resize} \
--gpu 1 \