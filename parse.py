import sys
import glob
import yaml
import pandas as pd

def parse_yamls_to_df(result_dir):
    # List to store data from each file
    data_list = []

    # Get all files matching the pattern
    files = glob.glob(f'{result_dir}/*.yml')

    dict_list = []
    for file in files:
        with open(file, 'r') as f:
            # Load the YAML content
            content = yaml.safe_load(f)
            
            # Extract the required data
            if 'percent_error_summary_statistics' in content and 'AllCandidate' in content['percent_error_summary_statistics']:
                data = dict()
                data['num_repeats'] = content['num_repeats']
                data['num_candidates'] = content['num_candidates']
                data['alpha'] = content['alpha']
                for candiate in content['candidate_vote']:
                    data.update(candiate)
                data.update(content['percent_error_summary_statistics']['AllCandidate'])
                # Convert the data to a DataFrame
                dict_list.append(data)
                
    # Convert the list of dictionaries to a DataFrame    
    df = pd.DataFrame(dict_list)
    
    return df

if __name__ == "__main__":
    result_dir = sys.argv[1]
    df = parse_yamls_to_df(result_dir)
    df.to_csv(f'{result_dir}/summary.csv', index=False)