#python3 -m src.train_quweit_lut_early_exit_g0_ce --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
 
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.log
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_2.log
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_3.log


#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.log


#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.log



#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.log



#python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.log

#python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.log





#python3 -m src.train_quweit_lut_early_exit_alt --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_1.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_1.log

#python3 -m src.train_quweit_lut_early_exit_alt --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_2.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_2.log
    
#python3 -m src.train_quweit_lut_early_exit_alt --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_3.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 
#python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_192_3.log
  



python3 -m src.train_quweit_lut_early_exit_alt_constrain --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_1.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 --min_exit_accs 0.92,0.93,0.95,0.98 --max_exit_rates 0.08,0.1,0.1,0.5 --min_final_rate 0.45 --no-train_classifier --thr 2.6581,3.0537,3.5199,8.0652

python3 -m src.train_quweit_lut_early_exit_alt_constrain --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_2.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 --min_exit_accs 0.92,0.93,0.95,0.98 --max_exit_rates 0.08,0.1,0.1,0.5 --min_final_rate 0.45 --no-train_classifier --thr 2.6581,3.0537,3.5199,8.0652

python3 -m src.train_quweit_lut_early_exit_alt_constrain --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_3.pth --final_train_layers 8,9,10,11 --train_exit_ids 2,4,6,8 --min_exit_accs 0.92,0.93,0.95,0.98 --max_exit_rates 0.08,0.1,0.1,0.5 --min_final_rate 0.45 --no-train_classifier --thr 2.6581,3.0537,3.5199,8.0652