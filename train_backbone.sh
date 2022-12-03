# backbone=mobilenet_v2
backbone=resnet50
# backbone=alexnet 
# backbone=squeezenet1_1

resize=224
lr=0.0002
epochs=40
batch_size=64

python main.py \
--backbone ${backbone} \
--lr ${lr} \
--epochs ${epochs} \
--batch_size ${batch_size} \
--resize ${resize} \
--gpu 1 \
--ovs \
--aug