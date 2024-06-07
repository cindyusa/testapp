import pandas as pd

def evaluate_model(gold_standard_file, extract_information):
    gold_standard = pd.read_csv(gold_standard_file)
    
    results = []
    for _, row in gold_standard.iterrows():
        sample_paragraph = row['paragraph']
        expected_death_benefit = row['death_benefit']
        expected_interest_rate = row['interest_rate']
        
        death_benefit, interest_rate = extract_information(sample_paragraph)
        
        result = {
            'sample_paragraph': sample_paragraph,
            'expected_death_benefit': expected_death_benefit,
            'extracted_death_benefit': death_benefit,
            'expected_interest_rate': expected_interest_rate,
            'extracted_interest_rate': interest_rate
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('evaluation_results.csv', index=False)
    return results_df