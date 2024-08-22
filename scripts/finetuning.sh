current_pth=$(pwd)
fairseq_pth=$current_pth/fairseq
src_pth=$current_pth/src

data_pth=???
checkpoint_save_pth=???

PYTHONPATH=/mnt/ssd3/jh/Exp/iclr24/av_hubert/fairseq \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-hydra-train \
    --config-dir $src_pth/conf/ \
    --config-name finetuning.yaml \
    task.data=$data_pth \
    task.label_dir=$data_pth \
    task.tokenizer_bpe_model=$current_pth/spm1000/spm_unigram1000.model \
    model.w2v_path=$src_pth/pretrained_models/mavhubert/mavhubert.pt \
    hydra.run.dir=$checkpoint_save_pth \
    common.user_dir=$src_pth