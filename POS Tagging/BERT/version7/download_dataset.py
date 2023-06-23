import datasets

kannada_datasets = []

# 
IndicCOPA = datasets.load_dataset('ai4bharat/IndicCOPA', 'translation-kn')
print("Sets in this corpus:", IndicCOPA.keys())
print("Columns in this corpus:", IndicCOPA['test'][0].keys())
print("Example:", IndicCOPA['test'][0])
kannada_datasets.append(IndicCOPA)

