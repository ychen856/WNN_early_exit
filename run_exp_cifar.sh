python3 -m src.train_quweit_lut_early_exit_g0_ce --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
 
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_1.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_2.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_ce_192_3.log


python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_1.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_2.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_192_3.log


python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_1.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_2.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_final --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_192_3.log



python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_1.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_2.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_kd_margin --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_1a_kd_final_margin_192_3.log



python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_1.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_2.log

python3 -m src.train_quweit_lut_early_exit_g0_1a_dep --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --path_out /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth --exit_layers 2,4,6,8 --epochs 50 --k 192,192,192,192
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_g0_a1_kd_dep_192_3.log
