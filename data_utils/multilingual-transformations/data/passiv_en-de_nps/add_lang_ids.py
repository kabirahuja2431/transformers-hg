import json
import sys

in_file = sys.argv[1]
out_file = sys.argv[1].split(".json")[0] + ".lang_id.json"

ENGLISH_DET = {"the", "a", "my", "our", "some", "your", "her"}
GERMAN_DET = {"die", "einige", "der", "mein", "meine", "meinem", 
              "den", "dein", "deine", "deinem", "unser", "unsere", "ein", "einem",
              "dem", "ihr", "ihre", "ihrem"}

with open(in_file, 'r') as in_data, open(out_file, 'w') as out_data:
    for example in in_data:
        json_obj = json.loads(example.strip())
        # identify source and target languages (will always be the same)
        lang_id = None
        src_words = json_obj["translation"]["src"].split()
        for word in src_words:
            if word in ENGLISH_DET:
                lang_id = "en_XX"
                break
            elif word in GERMAN_DET:
                lang_id = "de_DE"
                break
        if not lang_id:
            raise Exception(f"could not classify sentence: \"{src_words}\"")
        
        # add to JSON
        json_obj["translation"]["lang"] = lang_id
        out_data.write(json.dumps(json_obj) + "\n")
