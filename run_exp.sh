#python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_1.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_1.log

#python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_1.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_1.log

#python -m src.train_early_exit_g0_1a_kd_final --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_1.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_1.log

#python -m src.train_early_exit_g0_1a_kd_margin --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_1.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_1.log

#python3 -m src.cotrain_early_exit_alt --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_1.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_1.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_1.log

#python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_1.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_1.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_1.log



#python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_2.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_2.log

#python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_2.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_2.log

#python -m src.train_early_exit_g0_1a_kd_final --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_2.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_2.log

#python -m src.train_early_exit_g0_1a_kd_margin --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_2.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_2.log

#python3 -m src.cotrain_early_exit_alt --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_2.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_2.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_2.log

#python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_2.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_2.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_2.log



#python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_3.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_8000_3.log

#python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_3.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_3.log

#python -m src.train_early_exit_g0_1a_kd_final --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_3.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_final_8000_3.log

#python -m src.train_early_exit_g0_1a_kd_margin --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_3.pth --exit_layers "0,1" --k "8000,8000"
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_margin_8000_3.log

#python3 -m src.cotrain_early_exit_alt --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_3.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_3.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_8000_3.log

#python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_8000_3.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_3.pth --exit_layers "0,1" --k "8000,8000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
#python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_8000_3.log











python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_6000_1.pth --exit_layers "0,1" --k "6000,6000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_1.pth --exit_layers "0,1" --k "6000,6000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_1.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_1.pth --exit_layers "0,1" --k "6000,6000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_1.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_6000_2.pth --exit_layers "0,1" --k "6000,6000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_2.pth --exit_layers "0,1" --k "6000,6000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_2.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_2.pth --exit_layers "0,1" --k "6000,6000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_2.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_6000_3.pth --exit_layers "0,1" --k "6000,6000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_3.pth --exit_layers "0,1" --k "6000,6000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_6000_3.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_3.pth --exit_layers "0,1" --k "6000,6000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_6000_3.log




python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_4000_1.pth --exit_layers "0,1" --k "4000,4000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_1.pth --exit_layers "0,1" --k "4000,4000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_1.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_1.pth --exit_layers "0,1" --k "4000,4000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_1.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_4000_2.pth --exit_layers "0,1" --k "4000,4000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_2.pth --exit_layers "0,1" --k "4000,4000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_2.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_2.pth --exit_layers "0,1" --k "4000,4000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_2.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_4000_3.pth --exit_layers "0,1" --k "4000,4000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_3.pth --exit_layers "0,1" --k "4000,4000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_4000_3.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_3.pth --exit_layers "0,1" --k "4000,4000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_4000_3.log




python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_2000_1.pth --exit_layers "0,1" --k "2000,2000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_1.pth --exit_layers "0,1" --k "2000,2000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_1.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_1.pth --exit_layers "0,1" --k "2000,2000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_1.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_1.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_2000_2.pth --exit_layers "0,1" --k "2000,2000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_2.pth --exit_layers "0,1" --k "2000,2000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_2.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_2.pth --exit_layers "0,1" --k "2000,2000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_2.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_2.log

python -m src.train_early_exit_g0_1a_ce --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_ce_2000_3.pth --exit_layers "0,1" --k "2000,2000"
python -m src.train_early_exit_g0_1a_kd --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/wnn_FMNIST_backbone.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_3.pth --exit_layers "0,1" --k "2000,2000"
python3 -m src.cotrain_early_exit_alt_constrain --dataset FMNIST --backbone_ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_g0_1a_kd_2000_3.pth --path_out /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_3.pth --exit_layers "0,1" --k "2000,2000" --final_train_layers 1 --train_exit_ids 0,1 --min_exit_accs 0.98,0.98 --max_exit_rates 0.25,0.45 --min_final_rate 0.3 --no-train_classifier --thr 5.615,3.988 --sweep_selection_baseline_overall 87.63
python3 -m eval_scripts --ckpt /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_3.pth 2>&1 | tee /Users/yi-chunchen/workspace/WNN_early_exit/model/F_wnn_w_exit_FMNIST_alt_constrain_2000_3.log
