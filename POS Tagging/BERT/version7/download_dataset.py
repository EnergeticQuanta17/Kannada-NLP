import datasets

IndicCOPA = datasets.load_dataset('ai4bharat/IndicCOPA', 'translation-kn')

print(IndicCOPA.keys())
print(IndicCOPA['test'][0])
print(type(IndicCOPA))
