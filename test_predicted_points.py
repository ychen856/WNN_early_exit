

ROOT = Path("/Users/yi-chunchen/workspace/WNN_early_exit")
DEFAULT_DATA_ROOT = ROOT / "datasets"

RHO_100 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_8000_1.pth"
RHO_75 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_6000_1.pth"
RHO_50 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_4000_1.pth"
RHO_25 = ROOT / "model/F_wnn_w_exit_FMNIST_alt_constrain_2000_1.pth"

RHO_MODEL_PATHS = {
    100: RHO_100,
    75: RHO_75,
    50: RHO_50,
    25: RHO_25,
}


python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_1_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_2_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_192_3_fixed.log

python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_1_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_2_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_144_3_fixed.log

python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_1_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_2_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_96_3_fixed.log

python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_1.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_1_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_2.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_2_fixed.log
python3 -m eval_scripts --backbone_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u/best.pth --exit_ckpt /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_3.pth 2>&1 | tee /ychen-storage-fast/WNN_early_exit/model/weightless_all_v2_final_u_alt_constrain_48_3_fixed.log
 