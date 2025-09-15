device=1

# Dataset settings
cut=3
image_size=224

# Training settings
use_fp16=True
microbatch=12

# Architecture settings
attention_resolutions='28,14,7'

export CUDA_VISIBLE_DEVICES=$device
export OPENAI_LOGDIR="./models/cut_${cut}_imgsize_${image_size}_unconditional/"

MODEL_FLAGS="--attention_resolutions ${attention_resolutions} --class_cond False --image_size ${image_size} --learn_sigma True --num_channels 128 --num_heads 4 --num_res_blocks 2 --resblock_updown True --use_fp16 ${use_fp16} --use_scale_shift_norm True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-4 --batch_size 32 --microbatch ${microbatch} --lr_anneal_steps 150000 --use_condition False"
# If resume checkpoint needed, uncomment here and specify the checkpoint
#RESUME_CHECKPOINT="--resume_checkpoint ${OPENAI_LOGDIR}model${checkpoint}.pt"

INPUT_DIR="./datasets/Deli-CS/adjoint_tsmi/"
REF_DIR="./datasets/Deli-CS/reference_tsmi"

echo "python scripts/mrf_train_2d.py --input_dir ${INPUT_DIR} --ref_dir ${REF_DIR} --cut ${cut} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} ${RESUME_CHECKPOINT}"
      python scripts/mrf_train_2d.py --input_dir ${INPUT_DIR} --ref_dir ${REF_DIR} --cut ${cut} ${MODEL_FLAGS} ${DIFFUSION_FLAGS} ${TRAIN_FLAGS} ${RESUME_CHECKPOINT}
