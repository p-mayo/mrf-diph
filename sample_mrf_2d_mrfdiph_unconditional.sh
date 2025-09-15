device=0
cut=3
image_size=224
attention_resolutions='28,14,7'

dict_match=True
proximal=True
gamma_fixed=False
max_iters=5
timesteps=30
gamma=0.01
lambda=0.0001
xi=1.0
checkpoint=100000

export CUDA_VISIBLE_DEVICES=${device}
export OPENAI_LOGDIR="./models/cut_${cut}_imgsize_${image_size}_unconditional/mrfdiph_samples_lambda${lambda}"
MODEL_FLAGS="--attention_resolutions ${attention_resolutions} --class_cond False --image_size ${image_size} --learn_sigma True --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False"
SAMPLING_FLAGS="--timestep_respacing ${timesteps} --use_condition False --test_vols 0 --test_slices 5,10,15"
PROXOP_FLAGS="--lmbda ${lambda} --gamma ${gamma} --xi ${xi} --proxop_cg_maxiter ${max_iters} --gamma_fixed ${gamma_fixed} --dict_match ${dict_match} --proximal ${proximal}"

INPUT_DIR="./datasets/Deli-CS/adjoint_tsmi/"
REF_DIR="./datasets/Deli-CS/reference_tsmi"
QMAPS_REF_DIR="./datasets/Deli-CS/qmaps"
KSPACE_DIR="./datasets/Deli-CS/synthesized_ksp"
MODEL_PATH="./models/cut_${cut}_imgsize_${image_size}_unconditional/ema_0.9999_${checkpoint}.pt"

echo "python scripts/mrf_sample_mrfdiph.py --input_dir ${INPUT_DIR} --ref_dir ${REF_DIR} --qmaps_ref_dir ${QMAPS_REF_DIR} --kspace_dir ${KSPACE_DIR} --cut ${cut} --model_path ${MODEL_PATH} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${SAMPLING_FLAGS} ${PROXOP_FLAGS}"
      python scripts/mrf_sample_mrfdiph.py --input_dir ${INPUT_DIR} --ref_dir ${REF_DIR} --qmaps_ref_dir ${QMAPS_REF_DIR} --kspace_dir ${KSPACE_DIR} --cut ${cut} --model_path ${MODEL_PATH} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${SAMPLING_FLAGS} ${PROXOP_FLAGS}
