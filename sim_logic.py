import os
import sys
import yaml
import json
import random
import numpy as np
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

class SimulationParameters:
    def __init__(self, input_param):
        # parse the parameters
        self.name = input_param['sim_name']
        log.info(f'Simulation Name: {self.name}')
        
        self.result_file = input_param['result_file']
        log.info(f'Result File: {self.result_file}')
        
        result_dir = os.path.dirname(self.result_file)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        self.num_repeats = input_param['num_repeats']
        log.info(f'Number of repeats: {self.num_repeats}')
        
        if 'num_candidates' not in input_param:
            # If num_candidates is not provided, derive it from candidate_votes and imaginary_candidates
            self.candidate_votes = input_param['candidate_votes']
            self.imaginary_candidates = input_param['imaginary_candidate']
            if len(self.candidate_votes) != len(self.imaginary_candidates):
                raise ValueError(f'Number of candidate: {len(self.candidate_votes)} does not match number of imaginary candidates: {self.imaginary_candidates}')
            self.num_candidates = len(self.candidate_votes)
            self.candidates = [list(vote.keys())[0] for vote in self.candidate_votes]
            self.votes = [list(vote.values())[0] for vote in self.candidate_votes]
        else:
            # If num_candidates is provided, use it directly
            self.num_candidates = input_param['num_candidates']
            self.votes = input_param['candidate_votes']
            if len(self.votes) != self.num_candidates:
                raise ValueError(f'lenght of vote list: {len(self.votes)} does not match number of candidates: {self.num_candidates}')
            # name the candidates from c1 to cN and imaginary candidates from i1 to iN
            self.candidates = [f'c{i+1}' for i in range(self.num_candidates)]
            self.imaginary_candidates = [f'i{i+1}' for i in range(self.num_candidates)]
            self.candidate_votes = [{self.candidates[i]: self.votes[i]} for i in range(self.num_candidates)]
            
        log.info(f'Number of Candidates: {self.num_candidates}')
        log.info(f'Candidate Votes: {self.candidate_votes}')
        log.info(f'Imaginary Candidates: {self.imaginary_candidates}')
        
        # Calculate beta from alpha and ensure alpha is greater than beta
        self.alpha = input_param['alpha']
        if self.alpha >= 1 or self.alpha <= 0:
            raise ValueError(f'alpha: {self.alpha} must be between 0 and 1.')
        self.beta = (1 - self.alpha) / (self.num_candidates - 1)
        if self.alpha <= self.beta:
            raise ValueError(f'alpha: {self.alpha} must be greater than beta: {self.beta}')
        log.info(f'alpha: {self.alpha}')
        log.info(f'beta: {self.beta}')
        
        # Create the probability matrix given alpha and beta
        self.probability_matrix = np.full((self.num_candidates, self.num_candidates), self.beta)
        np.fill_diagonal(self.probability_matrix, self.alpha)
        log.info(f'Probability Matrix:\n {self.probability_matrix}')
        
        # Invert the probability matrix
        self.invert_probability_matrix = Simulation.invert_matrix(self.probability_matrix)
        log.info(f'Inverse Probablity matrix:\n {self.invert_probability_matrix}')
        
        # Calculate expected imaginary votes by multiplying the votes with the probability matrix
        self.expected_imaginary_votes = np.round(np.dot(self.votes, self.probability_matrix))
        log.info(f'Expected imaginary votes: {self.expected_imaginary_votes}')
        
        # This flag will be set to True after simulation summary is calculated
        self.has_simulation_summary = False
        
    def simulation_summary(self, data):
        # calculate percent error in the number of votes computed for each candidate over all simulation
        # and include the summary in the output
        # Also combined the percent error for all candidates and compute the summary of all candidates too
        self.summary = dict()
        combined = list()
        for i in range(self.num_candidates):
            # Calculate summary of percent error for each candidate
            percent_error = data[f'{self.candidates[i]}_percent_error']
            self.summary[f'{self.candidates[i]}'] = percent_error.describe().to_dict()
            combined.append(percent_error)
            
        # Combine percent errors for all candidates and calculate summary
        combined_data = pd.concat(combined, axis=0)
        self.summary['AllCandidate'] = combined_data.describe().to_dict()
        
        self.has_simulation_summary = True
        
    def save(self):
        param = dict()
        param['sim_name'] = self.name
        param['result_file'] = self.result_file
        param['num_repeats'] = self.num_repeats
        param['num_candidates'] = self.num_candidates
        param['alpha'] = self.alpha
        param['beta'] = self.beta
        param['candidate_vote'] = self.candidate_votes
        param['expected_imaginary_votes'] = {i: float(v) for i, v in zip(self.imaginary_candidates, self.expected_imaginary_votes)}
        param['probability'] = {self.candidates[i]: {self.imaginary_candidates[j]: round(float(self.probability_matrix[j][i]), 5) for j in range(self.num_candidates)} for i in range(self.num_candidates)}
        param['inverted_probability'] = {self.candidates[i]: {self.imaginary_candidates[j]: round(float(self.invert_probability_matrix[j][i]), 5) for j in range(self.num_candidates)} for i in range(self.num_candidates)}
        
        json_format = dict()
        json_format['candidates'] = json.dumps(self.candidates)
        json_format['votes'] = json.dumps(self.votes)
        json_format['imaginary_candidate'] = json.dumps(self.imaginary_candidates)
        json_format['expected_imaginary_vote'] = json.dumps(self.expected_imaginary_votes.tolist())
        json_format['probability'] = json.dumps(self.probability_matrix.tolist())
        json_format['inverted_probability'] = json.dumps(self.invert_probability_matrix.tolist())
        param['json'] = json_format
        
        if self.has_simulation_summary:
            param['percent_error_summary_statistics'] = self.summary
            
        # Save parameters and simulation summary to a YAML file
        with open(f'{self.result_file}.yml', 'w') as file:
            yaml.dump(param, file, sort_keys=False, indent=2, width=float("inf"))
            
class Simulation:
    
    @classmethod
    def invert_matrix(cls, matrix):
        try:
            # Attempt to invert the matrix
            inverse_matrix = np.linalg.inv(matrix)
            return inverse_matrix
        except np.linalg.LinAlgError:
            # Return an error message if the matrix is singular and cannot be inverted
            raise ValueError('Matrix is singular and cannot be inverted.')
        
    def __init__(self, input_param):
        # Initialize simulation parameters
        self.param = SimulationParameters(input_param)
        # Save the initial parameters into a yaml file
        self.param.save()
        
    def simulate(self): # pefrom a single simulation
        p = self.param
        # Initialize imaginary votes and crosstab votes
        self.imaginary_votes = [0] * p.num_candidates # number of votes for each imaginary candidate after applying randomness to the actual votes individually and based on probality matrix
        self.crosstab_votes = [[0] * p.num_candidates for _ in range(p.num_candidates)] # number of votes that each candidate contribute to the imaginary candidate after applying randomness and based on probability matrix
        
        # Simulate the voting process
        for candidate in range(p.num_candidates): # for each candidate
            for _ in range(p.votes[candidate]): # for each of the actual votes
                # generate a random number between 0 and 1
                random_value = random.random()
                
                # Transform the candidate based on the random value and probability matrix
                imaginary_candidate = self.transform(candidate, random_value)
                
                # Update the imaginary votes and crosstab votes
                self.imaginary_votes[imaginary_candidate] += 1
                self.crosstab_votes[imaginary_candidate][candidate] += 1
            
        # Compute the number of votes given the imaginary votes
        self.computed_votes = np.round(np.dot(self.imaginary_votes, p.invert_probability_matrix))
        
    def transform(self, candidate, random_value):
        for imaginary_candidate in range(self.param.num_candidates):
            probability = self.param.probability_matrix[imaginary_candidate][candidate]
            if random_value < probability:
                return imaginary_candidate
            else:
                random_value -= probability
        # Example: candidate is Alice, probability of Alice -> Red = 0.7 and probability of Alice -> Blue = 0.3
        # If random_value is 0.6, since 0.6 < 0.7, return Red
        # If random_value is 0.8, since 0.8 > 0.7, subtract 0.7 from 0.8 and remaining is 0.1
        # Since 0.1 < 0.3, return Blue
                
    def get_data(self):
        p = self.param
        data = dict()
        col_order = 0 # Used as a 9 digit prefix on column names to sort columns in the output data frame
        
        # Collect actual votes data
        for i in range(p.num_candidates):
            data[f'{col_order:09d}_{p.candidates[i]}_actual'] = p.votes[i]
            col_order += 1
            
        # Collect expected imaginary votes data
        for i in range(p.num_candidates):
            data[f'{col_order:09d}_{p.imaginary_candidates[i]}_expected'] = int(p.expected_imaginary_votes[i])
            col_order += 1
        
        # Collect crosstab votes data
        for i in range(p.num_candidates):
            for j in range(p.num_candidates):
                data[f'{col_order:09d}_{p.candidates[i]}_to_{p.imaginary_candidates[j]}'] = self.crosstab_votes[j][i]
                col_order += 1
            
        # Collect counted imaginary votes data
        for i in range(p.num_candidates):
            data[f'{col_order:09d}_{p.imaginary_candidates[i]}_counted'] = self.imaginary_votes[i]
            col_order += 1
            
        # Collect computed votes data
        for i in range(p.num_candidates):
            data[f'{col_order:09d}_{p.candidates[i]}_computed'] = int(self.computed_votes[i])
            col_order += 1
            
        # Collect percent errors for each candidate
        max_percent_error = 0
        for i in range(p.num_candidates):
            persent_error = abs(p.votes[i] - self.computed_votes[i]) * 100 / p.votes[i]
            if persent_error > max_percent_error:
                max_percent_error = persent_error
            data[f'{col_order:09d}_{p.candidates[i]}_percent_error'] = persent_error
            col_order += 1
        data[f'{col_order:09d}_max_percent_error'] = max_percent_error

        return data
    
    def run(self):
        results = [] # List to store the results of each simulation
        log.info(f'Starting Simulation: {self.param.name} ({self.param.num_repeats} repeats)')
        for i in range(self.param.num_repeats):
            log.info(f'>>> Running Simulation: {i + 1} out of {self.param.num_repeats} ({self.param.name})')
            self.simulate()
            result = self.get_data() # Get the result from each simulation
            results.append(result)
        log.info(f'Simulation {self.param.name} completed.')
        
        # Create a DataFrame from the results
        data = pd.DataFrame(results)
        # Sort columns in the output data frame and remove the 9 digit prefix used for sorting
        data = data.reindex(sorted(data.columns), axis=1)
        data.columns = [col[10:] for col in data.columns]
        # Add an index column named 'sim_index'
        data.index.name = 'sim_index'
        # Save the results to a CSV file
        data.to_csv(f'{self.param.result_file}.csv')
        
        # Add simulation summary to the param file and save it (overwrite)
        self.param.simulation_summary(data)
        self.param.save()

def main():
    """
    Main function to run the simulation.
    This function expects a single command-line argument specifying the path to a YAML parameter file.
    It performs the following steps:
    1. Checks if the correct number of command-line arguments is provided.
    2. Logs the parameter file path.
    3. Loads parameters from the specified YAML file.
    4. For each parameter set in the YAML file, creates a Simulation object and runs the simulation.
    Raises:
        FileNotFoundError: If the specified parameter file does not exist.
        ValueError: If there is an error decoding the YAML file.
    """
    if len(sys.argv) != 2:
        log.error('Usage: python sim.py param_file.yml')
        sys.exit(1)
    log.info(f'Parameter File: {sys.argv[1]}')
    parameter_file = sys.argv[1]

    # load the parameter from yaml file
    try:
        with open(parameter_file, 'r') as file:
            input_param = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f'File {parameter_file} not found.')
    except yaml.YAMLError:
        raise ValueError(f'Error decoding YAML from file {parameter_file}.')
       
    for param in input_param:
        print("====================================================================================================")
        sim = Simulation(param)
        sim.run()

if __name__ == '__main__':
    main()