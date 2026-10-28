"""Medication lexicon for the clinical-cue review.

Built from the three corpora themselves rather than from an external terminology service, which was
not available. Two parts: drug names found by morphological suffix or by an explicit common-drug
list, and medication class terms a clinician would treat as equally consequential if lost
("antibiotics", "inhaler", "steroid"). Spelling variants and transcript misspellings are kept,
because the reference transcript is the gold standard and the detector must match what it says.
"""

DRUG_NAMES = set("""
acetaminophen advil amitriptyline amlodipine amoxicillin amoxycillin apixaban artemether aspirin
atorvastatin azithromycin benadryl bisoprolol buscopan calpol cetirizine chloroquine citalopram
claritin clenil clindamycin clopidogrel coartem codeine cotrimoxazole dexamethasone digoxin
dioralyte empagliflozin flagyl fluoxetine fostair furosemide gabapentin gaviscon hydromorphone
insulin lansoprazole laxido lemsip levothyroxine lisinopril lisonopril loratadine metformin
metoprolol metronidazole morphine movicol multivitamin naproxen nurofen omeprazole oxycodone
paludrine panadol pantoprazole paracetamol penicillin piriton pregabalin prednisolone prednisone
quinine rabeprazole ramipril ranitidine rivaroxaban rosuvastatin salbutamol senna septrin seretide
sertraline simvastatin spironolactone sudafed symbicort telmisartan tramadol tylenol ventolin
warfarin zinopril zyrtec
""".split())

CLASS_TERMS = set("""
antibiotic antibiotics antihistamine antihistamines inhaler inhalers nebuliser nebulizer
painkiller painkillers puffer steroid steroids
""".split())

MED_CUES = DRUG_NAMES | CLASS_TERMS

if __name__ == "__main__":
    print("drug names : {}".format(len(DRUG_NAMES)))
    print("class terms: {}".format(len(CLASS_TERMS)))
    print("total      : {}".format(len(MED_CUES)))
