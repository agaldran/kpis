python train_seg.py --im_size 1024/1024 --csv_path_tr data/tr_f1.csv --save_path f1/fpn_mitb_4_1024_ce_dice --model_name fpn_mitb_4 --loss2 dice --alpha2 1.0 --batch_size 4
python train_seg.py --im_size 1024/1024 --csv_path_tr data/tr_f2.csv --save_path f2/fpn_mitb_2_1024_ce_dice --model_name fpn_mitb_4 --loss2 dice --alpha2 1.0 --batch_size 4
python train_seg.py --im_size 1024/1024 --csv_path_tr data/tr_f4.csv --save_path f4/fpn_mitb_2_1024_ce_dice --model_name fpn_mitb_4 --loss2 dice --alpha2 1.0 --batch_size 4
python train_seg.py --im_size 1024/1024 --csv_path_tr data/tr_f4.csv --save_path f4/fpn_mitb_2_1024_ce_dice --model_name fpn_mitb_4 --loss2 dice --alpha2 1.0 --batch_size 4
python train_seg.py --im_size 1024/1024 --csv_path_tr data/tr_f5.csv --save_path f5/fpn_mitb_2_1024_ce_dice --model_name fpn_mitb_4 --loss2 dice --alpha2 1.0 --batch_size 4


