import datasets

kannada_datasets = []

# The dataset consists of examples for the Choice of Plausible Alternatives (COPA) task, where each example includes a premise, two choices, a question, a label indicating the correct choice, an index, and a field indicating if any changes were made.
IndicCOPA = datasets.load_dataset('ai4bharat/IndicCOPA', 'translation-kn')
print("Sets in this corpus:", list(IndicCOPA.keys()))
print("Columns in this corpus:", list(IndicCOPA['test'][0].keys()))
print("Example:", IndicCOPA['test'][0])
kannada_datasets.append(IndicCOPA)  