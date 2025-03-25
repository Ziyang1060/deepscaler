# set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

# Default values
MODEL_PATH="$HOME/DeepScaleR-1.5B-Preview"
# Possible values: aime, amc, math, minerva, olympiad_bench
DATATYPES=("aime")
OUTPUT_DIR="$HOME"  # Add default output directory

# Parse named arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --datasets)
            # Convert space-separated arguments into array
            shift
            DATATYPES=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                DATATYPES+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --model <model_path> --datasets dataset1 dataset2 ... --output-dir <output_directory>"
            exit 1
            ;;
    esac
done

# Echo the values for verification
echo "Model Path: ${MODEL_PATH}"
echo "Datasets: ${DATATYPES[@]}"
echo "Output Directory: ${OUTPUT_DIR}"

# Loop through all datatypes
for DATA_TYPE in "${DATATYPES[@]}"; do
    python3 -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$HOME/deepscaler/data/${DATA_TYPE}.parquet \
        data.output_path=${OUTPUT_DIR}/${DATA_TYPE}.parquet \
        data.n_samples=32 \
        data.batch_size=2048 \
        model.path=${MODEL_PATH} \
        rollout.temperature=0.6 \
        rollout.response_length=32768 \
        rollout.top_k=-1 \
        rollout.top_p=0.95 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.tensor_model_parallel_size=1
done

# aime, amc, math, minerva, olympiad_bench

# Test
# bash scripts/eval/eval_model.sh --model /processing_data/search/zengziyang/DeepScaleR-1.5B-Preview --datasets aime aime25 --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepScaleR-1.5B-Preview

# Official
# ( bash scripts/eval/eval_model.sh --model /model_load/DeepSeek-R1-Distill-Qwen-1.5B --datasets aime aime25 math minerva olympiad_bench --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-1.5B &)
# ( bash scripts/eval/eval_model.sh --model /model_load/DeepSeek-R1-Distill-Qwen-7B --datasets aime aime25 amc math minerva olympiad_bench --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-7B &)


# ( bash scripts/eval/eval_model.sh --model /model_load/DeepSeek-R1-Distill-Qwen-1.5B --datasets olympiad_bench --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-1.5B &)
# ( bash scripts/eval/eval_model.sh --model /model_load/DeepSeek-R1-Distill-Qwen-7B --datasets olympiad_bench --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-7B &)


# bash scripts/eval/eval_model.sh --model /model_load/DeepSeek-R1-Distill-Qwen-1.5B --datasets amc --output-dir /data_train/search/zengziyang/projects/deepscaler/outputs/DeepSeek-R1-Distill-Qwen-1.5B