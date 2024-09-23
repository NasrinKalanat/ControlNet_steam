CKPT_PATH=$1
OUTPUT_CKPT=$2

python tool_add_control.py "$CKPT_PATH" "$OUTPUT_CKPT"

rm -rf controlnet