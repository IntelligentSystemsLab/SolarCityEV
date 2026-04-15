#!/bin/bash

# Script to run the meta-learning training
# Usage: ./run.sh [options]

# Default values
CITY_ARGS=("London")
EPOCHS=2
SUPPORT_EPOCHS=5
CUSTOM_EPOCHS=5
LR=0.005
DIVIDE_MODE_ARGS=("by_month")
FOLDER_PATH="charging_data/by_station"
SEED=2023
BATCH_SIZE=""
PRINT_DETAILS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --city)
            shift
            CITY_ARGS=()
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                CITY_ARGS+=("$1")
                shift
            done
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --support_epochs)
            SUPPORT_EPOCHS="$2"
            shift 2
            ;;
        --custom_epochs)
            CUSTOM_EPOCHS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --divide_mode)
            shift
            DIVIDE_MODE_ARGS=()
            while [[ $# -gt 0 ]] && [[ ! $1 =~ ^-- ]]; do
                DIVIDE_MODE_ARGS+=("$1")
                shift
            done
            ;;
        --folder_path)
            FOLDER_PATH="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --print_details)
            PRINT_DETAILS="--print_details"
            shift
            ;;
        -h|--help)
            echo "Usage: ./run.sh [options]"
            echo ""
            echo "Options:"
            echo "  --city CITY [CITY ...]   City name(s) in English (default: London)"
            echo "                           Can specify multiple cities: --city London Paris NewYork"
            echo "  --epochs N               Number of training epochs (default: 300)"
            echo "  --support_epochs N       Number of support epochs (default: 5)"
            echo "  --custom_epochs N        Number of custom epochs (default: 5)"
            echo "  --lr RATE                Learning rate (default: 0.005)"
            echo "  --divide_mode MODE [MODE ...]  Data division mode(s) (default: by_month)"
            echo "                           Can specify multiple modes: --divide_mode by_month by_day"
            echo "  --folder_path PATH       Data folder path (default: charging_data/by_station)"
            echo "  --seed N                 Random seed (default: 2023)"
            echo "  --batch_size N           Batch size (default: None)"
            echo "  --print_details          Print detailed training information"
            echo "  -h, --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run.sh --city London --epochs 100 --lr 0.001"
            echo "  ./run.sh --city London Paris NewYork --divide_mode by_month by_day"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Build command
CMD="python code/run.py --city ${CITY_ARGS[@]} --epochs $EPOCHS --support_epochs $SUPPORT_EPOCHS --custom_epochs $CUSTOM_EPOCHS --lr $LR --divide_mode ${DIVIDE_MODE_ARGS[@]} --folder_path $FOLDER_PATH --seed $SEED"

if [ -n "$BATCH_SIZE" ]; then
    CMD="$CMD --batch_size $BATCH_SIZE"
fi

if [ -n "$PRINT_DETAILS" ]; then
    CMD="$CMD $PRINT_DETAILS"
fi

# Run the command
echo "Running: $CMD"
eval $CMD

