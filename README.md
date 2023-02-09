# UnsuperPoint_PyTorch
A pytorch version about unsuperpoint based on the following paper: https://arxiv.org/abs/1907.04011

@RPFey and @791136190 implementation：
https://github.com/RPFey/UnsuperPoint
https://github.com/791136190/UnsuperPoint_PyTorch

# How to install UnsuperPoint

    pip install -r requirements.txt

# How to train UnsuperPoint
To train unsuperpoint run the following command:

    python train.py --cfg_file Unsuper/configs/UnsuperPoint_coco.yaml --workers 4 --ckpt_save_interval 5

# How to export Predictions with UnsuperPoint

    python export.py --cfg_file Unsuper/configs/UnsuperPoint_coco.yaml --workers 4 --batch_size 16 --ckpt_dir output/ckpt --save_to_file

Where in UnsuperPoint_coco.yaml file you have to define the "train_name" which is the class that manages the dataset preprocessing while in "export_name", the sym linked folder is defined.

# Notes
The code implements the pytorch version of UnsuperPoint. Some of the parts  that cannot be successfully trained were modified. The actual test in the slam data set can reach to the effect of the orb.
The implementation in the code is not completely consistent with the content of the paper. The paper uses the correlation coefficient of each bit of the descriptor to supervise the expression ability of the descriptor. The code directly uses the characteristics of the descriptors in different positions.

***The users of this repo are free to contribute and improve the UnsuperPoint accuracy.***

# Reproduce notes
Youdao Cloud: http://note.youdao.com/s/IQBrgPio
