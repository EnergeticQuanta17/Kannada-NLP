import datasets

IndicCOPA = datasets.load_dataset('ai4bharat/IndicCOPA', 'translation-kn')

print(IndicCOPA.keys())
print(IndicCOPA['train'][0])
print(type(IndicCOPA))