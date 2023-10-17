import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules

# leitura de arquivo em excel
df = pd.read_excel('./Base/baseAula.xlsx')

#declarando as métricas mínimas
support_threshold = 0.14
confidence_threshold = 0.28

#declaração de algoritmos que iremos usar
algorithms = [apriori, fpgrowth]

# aplicação de métodos
for algorithm in algorithms:
    frequent_itemsets = algorithm(df, min_support=support_threshold, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=confidence_threshold)
    
    rules['lift'] = rules['lift']
    rules['leverage'] = rules['leverage']
    rules['conviction'] = rules['conviction']
    rules['zhang'] = rules['support'] - rules['antecedent support'] * rules['consequent support']
    
    algorithm_name = str(algorithm.__name__).capitalize()  
    print(f"Associações ({algorithm_name}):")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'leverage', 'conviction', 'zhang']])
