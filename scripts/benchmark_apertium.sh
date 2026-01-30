#!/bin/bash

# Declare language pairs used in apertium
declare -A LANG=(
   ["spa"]="es-eo" ["epo"]="eo-en" ["cat"]="ca-eo" ["eng"]="en-eo" 
)

for lang in "${!LANG[@]}"; do
    echo "Translating ${lang}..."
    start=`date +%s`
    langpair=${LANG[$lang]}
    source_file="../../../Downloads/flores200_dataset/devtest/${lang}_Latn.devtest"
    target_file="$langpair.apertium"

    # Check if target file exists
    if [ -f "$target_file" ]; then
        # Get line counts
        source_lines=$(wc -l < "$source_file")
        target_lines=$(wc -l < "$target_file")

        if [ "$source_lines" -eq "$target_lines" ]; then
            echo "Translation already complete for ${lang}. Skipping..."
            continue
        else
            echo "Resuming translation for ${lang} from line $((target_lines + 1))..."
        fi
    else
        target_lines=0
    fi

    # Continue or start translation
    tail -n +$((target_lines + 1)) "$source_file" | while IFS= read -r line; do
        echo "$line" | apertium $langpair >> "$target_file"
    done

    end=`date +%s`
    runtime=$((end-start))
    echo "Translation runtime for ${lang}: $runtime seconds"
done



