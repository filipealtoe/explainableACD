#!/bin/bash
# Train all architecture variants for diverse ensemble
# Usage: bash train_all_architectures.sh ~/data ~/results

DATA_DIR=${1:-~/data}
OUTPUT_BASE=${2:-~/arch_results}

echo "========================================"
echo "Training Multiple Architectures"
echo "========================================"
echo "Data dir: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo ""

# Common flags for best config
FLAGS="--focal-loss --llrd --rdrop --fgm --cosine-schedule --eval-on-devtest"

# Models to train
declare -a MODELS=("roberta-large" "bge-large-en-v1.5" "e5-large-v2" "gte-large")

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: $MODEL"
    echo "========================================"

    OUTPUT_DIR="$OUTPUT_BASE/${MODEL}"

    # Check if already trained
    if [ -f "$OUTPUT_DIR/results.json" ] || [ -f "$OUTPUT_DIR/$MODEL/results.json" ]; then
        echo "✓ $MODEL already trained, skipping..."
        continue
    fi

    python finetune_deberta_multimodel.py \
        --model "$MODEL" \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $FLAGS

    if [ $? -eq 0 ]; then
        echo "✓ $MODEL training complete"
    else
        echo "✗ $MODEL training failed"
    fi
done

echo ""
echo "========================================"
echo "All training complete!"
echo "========================================"
echo ""
echo "Next step: Run the multi-architecture ensemble:"
echo "python ensemble_multi_arch.py --data-dir $DATA_DIR --arch-dir $OUTPUT_BASE --deberta-dir ~/ensemble_results"
