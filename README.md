# kcg-ml-n-grams  

## Simple Prompt Generator  
Preprocess the civitai csv file first
```
$ python scripts/preprocess_civitai_csv.py \
    --data_csv_path <civitai csv file with phrases>
    --output_path <output path to save processed csv file>
```

```
$ python scripts/generate_prompts.py  \
    --data_csv_path <processed civitai csv file. must have probability and log probability columns> \
    --n_prompt <number of prompts to generate> \
    --save_path <msgpack save path>
```

## Scoring
Now supports only positive embedding scoring using linear and elm models
```
$ python scripts/score_random_prompts.py \
    --prompt_path <generated prompts msgpack save path> \
    --data_csv_path <civitai csv file with phrases> \
    --linear_model_weights <linear ranking model weights path> \
    --elm_model_weights <elm ranking model weights path> \
    --results_save_path <results save path>
```