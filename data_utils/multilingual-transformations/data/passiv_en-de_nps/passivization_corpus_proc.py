import random
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', help='Size for the training set', type=int, default=100000)
parser.add_argument('--dev_size', help='Size for the dev set', type=int, default=1000)
parser.add_argument('--test_size', help='Size for the test set', type=int, default=10000)
parser.add_argument('--gen_size', help='Size for the generalization set', type=int, default=10000)
parser.add_argument('--sentences', help='File of sentences to process', type=str, default=None)
parser.add_argument('--output_prefix', help='Prefix for output file names', type=str, default=None)
args = parser.parse_args()


fi = open(args.sentences, "r")


fo_train = open(args.output_prefix +  ".train", "w")
fo_dev = open(args.output_prefix +  ".dev", "w")
fo_test = open(args.output_prefix + ".test", "w")
fo_gen1 = open(args.output_prefix + ".gen_pp_s", "w")
fo_gen2 = open(args.output_prefix + ".gen_pp_o", "w")


DAT_DET_S_MAPPING = {
  "die": "der",
  "eine": "einer",
  "meine": "meiner",
  "deine": "deiner",
  "unsere": "unserer",
  "ihre": "ihrer"
}

DAT_DET_P_MAPPING = {
  "die": "den",
  "einige": "einigen",
  "meine": "meinen",
  "deine": "deinen",
  "unsere": "unseren",
  "ihre": "ihren"
  
}

VERB_PARTICIPLE_MAPPING = {
  "unterhielt": "unterhalten",
  "amüsierte": "amüsiert",
  "nervte": "genervt",
  "erfreute": "erfreut",
  "verwirrte": "verwirrt",
  "bewunderte": "bewundert",
  "akzeptierte": "akzeptiert",
  "bedauerte": "bedauert",
  "tröstete": "getröstet",
  "unterhielten": "unterhalten",
  "amüsierten": "amüsiert",
  "nervten": "genervt",
  "erfreuten": "erfreut",
  "verwirrten": "verwirrt",
  "bewunderten": "bewundert",
  "akzeptierten": "akzeptiert",
  "bedauerten": "bedauert",
  "trösteten": "getröstet"
}


# Words to delete from the sentences being processed
delList = ["N1S", "N1P", "N2", "N3", "N4", "N5", "N5S", "N5P", "N6", "N7", "N8", "DSA", "DPA", "DSN", "DPN",
        "VI1", "VI2", "VI3", "VT1", "VT2", "VT3", "VT4", "VT5", "VI", "VT", "RA", "RN"]
delDict = {}
for item in delList:
    delDict[item] = 1
DAT_DET_S_MAPPING = {
  "der": "dem",
  "ein": "einem",
  "mein": "meinem",
  "dein": "deinem",
  "unser": "unserem",
  "ihr": "ihrem"
}

DAT_DET_P_MAPPING = {
  "die": "den",
  "einige": "einigen",
  "meine": "meinen",
  "deine": "deinen",
  "unsere": "unseren",
  "ihre": "ihren"
  
}


# accusative -> nominative determiner mapping (sing)
NOM_DET_S_MAPPING = {
  "den": "der",
  "einen": "ein",
  "meinen": "mein",
  "deinen": "dein",
  "unseren": "unser",
  "ihren": "ihr"
}

# accusative -> nominative determiner mapping (plur)
NOM_DET_P_MAPPING = {
  "die"    : "die"     ,
  "einige": "einige"  ,
  "meine" : "meine"   ,
  "deine" : "deine"   ,
  "unsere": "unsere"  ,
  "ihre":   "ihre"
}

# nominative -> dative noun mapping (sing)
DAT_NOUN_S_MAPPING = {
  "Molch": "Molch",
  "Löwe": "Löwen",
  "Pfau": "Pfau",
  "Kater": "Kater",
  "Rabe": "Raben",
  "Salamander": "Salamander",
  "Dinosaurier": "Dinosaurier",
  "Papagei": "Papagei",
  "Geier": "Geier",
  "Wellensittich": "Wellensittich",
  "Esel": "Esel",
  "Hund": "Hund",
  "Ziesel": "Ziesel",
  "Molch": "Molch"
}


# nominative -> dative noun mapping (plur)
DAT_NOUN_P_MAPPING = {
  "Molche": "Molchen",
  "Löwen": "Löwen",
  "Pfaue": "Pfauen",
  "Kater": "Katern",
  "Raben": "Raben",
  "Salamander": "Salamandern",
  "Dinosaurier": "Dinosauriern",
  "Papageie": "Papageien",
  "Geier": "Geiern",
  "Wellensittiche": "Wellensittichen",
  "Esel": "Eseln",
  "Hunde": "Hunden",
  "Ziesel": "Zieseln"
}



# accusative -> nominative noun mapping (sing)
NOM_NOUN_S_MAPPING = {
  "Molch": "Molch",
  "Löwen": "Löwe",
  "Pfau": "Pfau",
  "Kater": "Kater",
  "Raben": "Rabe",
  "Salamander": "Salamander",
  "Dinosaurier": "Dinosaurier",
  "Papagei": "Papagei",
  "Geier": "Geier",
  "Wellensittich": "Wellensittich",
  "Esel": "Esel",
  "Hund": "Hund",
  "Ziesel": "Ziesel",
  "Molch": "Molch"
}


# accusative -> nominative noun mapping (plur)
NOM_NOUN_P_MAPPING = {
  "Molche": "Molche",
  "Löwen": "Löwen",
  "Pfaue": "Pfaue",
  "Kater": "Kater",
  "Raben": "Raben",
  "Salamander": "Salamander",
  "Dinosaurier": "Dinosaurier",
  "Papageie": "Papageie",
  "Geier": "Geier",
  "Wellensittichn": "Wellensittiche",
  "Esel": "Esel",
  "Hunde": "Hunde",
  "Ziesel": "Ziesel"
}

VERB_PARTICIPLE_MAPPING = {
  "unterhielt": "unterhalten",
  "amüsierte": "amüsiert",
  "nervte": "genervt",
  "erfreute": "erfreut",
  "verwirrte": "verwirrt",
  "bewunderte": "bewundert",
  "akzeptierte": "akzeptiert",
  "bedauerte": "bedauert",
  "tröstete": "getröstet",
  "unterhielten": "unterhalten",
  "amüsierten": "amüsiert",
  "nervten": "genervt",
  "erfreuten": "erfreut",
  "verwirrten": "verwirrt",
  "bewunderten": "bewundert",
  "akzeptierten": "akzeptiert",
  "bedauerten": "bedauert",
  "trösteten": "getröstet"
}


# Words to delete from the sentences being processed
delList = ["N1S", "N1P", "N2", "N3", "N4", "N5", "N5S", "N5P", "N6", "N7", "N8", "DSA", "DPA", "DSN", "DPN",
        "VI1", "VI2", "VI3", "VT1", "VT2", "VT3", "VT4", "VT5", "VI", "VT", "RA", "RN"]
delDict = {}
for item in delList:
    delDict[item] = 1

# convert sentence from active to passive
def passivize(sent):
    newSent = [s for s in sent]
    if "N5S" in sent:
        link_aux = "wurde"
        obj_ind = sent.index("N5S")
        newSent[obj_ind-1] = NOM_DET_S_MAPPING[sent[obj_ind-1]] 
        
    else:
        link_aux = "wurden"
        obj_ind = sent.index("N5P")
        newSent[obj_ind-1] = NOM_DET_P_MAPPING[sent[obj_ind-1]] 
        
    verb_ind = sent.index("VT3")

    participle = VERB_PARTICIPLE_MAPPING[sent[verb_ind + 1]]
    if sent[1] == "N1S":
      newSent[0] = DAT_DET_S_MAPPING[sent[0]]
      newSent[2] = DAT_NOUN_S_MAPPING[sent[2]]
    else:
      newSent[0] = DAT_DET_P_MAPPING[sent[0]]
      newSent[2] = DAT_NOUN_P_MAPPING[sent[2]]
      

    newSent = newSent[obj_ind - 1:-2] + [link_aux] + ["von"] + newSent[0:verb_ind] + [participle] + newSent[-2:]
    return newSent


def process(sent):
    if sent[-1] == "passiv":
        passiv = 1
    else:
        passiv = 0

    newSent = []
    for word in sent:
        if word not in delDict:
            newSent.append(word)
    
    newSentStr = " ".join(newSent[:-1])
    return newSentStr.replace(", .", ".")


def is_pp_on_subj(words):
  if words[5] == "N2":
    return True
  return False

def is_pp_on_obj(words):
  if "N2" in words[6:]:
    return True


def get_nouns(words):
  noun_set = []
  for i, word in enumerate(words):
    if word.startswith("N"):
      noun_set.append(words[i+1])
  
  return noun_set

# Sentences we've already used (to avoid repeats)
used_dict = {}

count_train = 0
count_dev = 0
count_test = 0
count_gen = 0

count_iid = 0
count_ood1 = 0
count_ood2 = 0

iid_examples = []
ood_examples1 = []
ood_examples2 = []

filter_counts = Counter()

for line in fi:
    if count_iid >= args.train_size + args.dev_size + args.test_size and count_ood1 >= args.gen_size and count_ood2 >= args.gen_size:
        break

    sent = line.strip()
    if sent in used_dict:
        continue

    used_dict[sent] = 1

    words = sent.split()
    
    nouns = get_nouns(words)
    
    
    #if len(nouns) > len(set(nouns)):
    #  filter_counts[len(nouns)] += 1
    #  continue

    pp_on_subj = is_pp_on_subj(words)
    
    pp_on_obj = is_pp_on_obj(words)
    passiv = random.getrandbits(1)

    if passiv:
        words.append("passiv")
    else:
        words.append("decl")

    if passiv:
        result = process(words) + " passiv" + "\t" + process(passivize(words)) + "\n"
    else:
        result = process(words) + " decl" + "\t" + process(words) + "\n"

    if (not passiv) and count_iid < args.train_size + args.dev_size + args.test_size:
        iid_examples.append(result)
        count_iid += 1
    elif pp_on_subj and passiv and count_ood1 < args.gen_size:
      ood_examples1.append(result)
      count_ood1 += 1
   
    elif pp_on_obj and passiv and count_ood2 < args.gen_size:
      ood_examples2.append(result)
      count_ood2 += 1
    

random.shuffle(iid_examples)
random.shuffle(ood_examples1)
random.shuffle(ood_examples2)

train_set = iid_examples[:args.train_size]
dev_set = iid_examples[args.train_size:args.train_size+args.dev_size]
test_set = iid_examples[args.train_size+args.dev_size:args.train_size+args.dev_size+args.test_size]

gen_set1 = ood_examples1[:args.gen_size]
gen_set2 = ood_examples2[:args.gen_size]

for elt in train_set:
    fo_train.write(elt)
for elt in dev_set:
    fo_dev.write(elt)
for elt in test_set:
    fo_test.write(elt)
for elt in gen_set1:
    fo_gen1.write(elt)
for elt in gen_set2:
    fo_gen2.write(elt)


