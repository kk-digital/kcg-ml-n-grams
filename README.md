# kcg-ml-n-grams  

## Simple Prompt Generator  
```
$ python scripts/generate_prompts.py  \
    --data_csv_path <civitai csv file with phrases. must have probability and log probability columns> \
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