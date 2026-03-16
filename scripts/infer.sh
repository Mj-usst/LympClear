cd /home/huawei/project/nnUNet/Data_preprocessing

python3 dicomToNii_No_Nrrd.py 

nnUNetv2_predict -d Dataset112_Vein -i /home/huawei/project/dataset/infer_nii/result -o /home/huawei/project/dataset/infer_result -f 3 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans 

python3 MIP.py

python3 GIF.py

# nnUNetv2_predict -d Dataset113_Vein_all_5710 -i /home/huawei/project/dataset/infer_nii/result -o /home/huawei/project/dataset/infer_result -f 3 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -device cuda:1
