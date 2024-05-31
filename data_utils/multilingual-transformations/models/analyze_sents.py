import sys

# QUESTION_AUXILIARIES = set(["has", "have", "is", "are", "can"])
QUESTION_AUXILIARIES = set(["has", "have"])

with open(sys.argv[1], 'r') as sents:
    sim = 0
    sent_1 = ""
    sent_2 = ""
    read_sent_1 = True
    for line in sents:
        if line.startswith("jaccard"):
            sim = float(line.strip().split(":")[1])
            continue
        if line.startswith("*-*-*-"):
            read_sent_1 = True
            # check sentences
            if sent_1.split()[0] in QUESTION_AUXILIARIES or sent_2.split()[0] in QUESTION_AUXILIARIES:
                if sent_1.split()[0] not in QUESTION_AUXILIARIES or sent_2.split()[0] not in QUESTION_AUXILIARIES:
                    print(f"{sent_1}\n----------\n{sent_2}\njaccard_sim: {sim}\n*-*-*-*-*-")
            
            sent_1 = ""
            sent_2 = ""
            continue

        if line.startswith("--"):
            read_sent_1 = False
            continue

        # otherwise, we're reading text
        if read_sent_1:
            sent_1 += line.strip() + " "
        else:
            sent_2 += line.strip() + " "

