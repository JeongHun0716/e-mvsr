languages_list=("en" "es" "fr" "it" "pt")


for target_lang in "${languages_list[@]}"
do
    beam_size=20
    lenpen=0
    current_pth=$(pwd)
    fairseq_pth=$current_pth/fairseq
    src_pth=$current_pth/src
    pretrained_model_pth=$src_pth/pretrained_models/finetuned/finetuned.pt
    label_pth=$current_pth/labels/$target_lang


    results_save_pth=$src_pth/outputs/$target_lang

    PYTHONPATH=$fairseq_pth CUDA_VISIBLE_DEVICES=3 python -B $src_pth/infer_s2s.py \
        --config-dir $src_pth/conf/ \
        --config-name s2s_decode.yaml \
        dataset.gen_subset=test \
        common_eval.path=${pretrained_model_pth} \
        common_eval.results_path=${results_save_pth} \
        override.modalities=['video'] \
        common.user_dir=${src_pth} \
        generation.beam=${beam_size} \
        generation.lenpen=${lenpen} \
        +override.data=${label_pth} \
        +override.label_dir=${label_pth}
done