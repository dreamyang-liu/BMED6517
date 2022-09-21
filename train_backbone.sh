backbone=resnet50
pretrained=True

lr=0.001
input_channel=1
epochs=100
batch_size=32


save_path=../model/${backbone}_pretrained_${pretrained}_lr_${lr}_input_channel_${input_channel}_epochs_${epochs}_batch_size_${batch_size}.pth

python main.py \
--backbone ${backbone} \
--pretrained ${pretrained} \
--lr ${lr} \
--input_channel ${input_channel} \
--epochs ${epochs} \
--batch_size ${batch_size} \
--save_path ${save_path}