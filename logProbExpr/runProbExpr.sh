for i in {1}; do
    echo "Expr Num: $i"
    echo "========== STARTING NEW EXPERIMENT ===================="

    rm -rf ./data
    rm -rf ./lora_weights
    python3 ./refactored/main.py

done