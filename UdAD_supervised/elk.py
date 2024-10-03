import pickle


with open('HCP_train_list.pickle', 'rb') as f:
    train_list = pickle.load(f)

print(train_list)