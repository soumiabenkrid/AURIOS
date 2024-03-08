import pickle

# Open the .pickle file in binary mode
with open('experiment_results/ID_TPCH_Test_Experiment/experiment_object.pickle', 'rb') as file:
    # Load the data from the file
    data = pickle.load(file)
    print(data)

