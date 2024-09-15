import torch

def get_entropy_of_dataset(tensor : torch.Tensor):
    label_column = [t[-1].tolist() for t in tensor] #extracting the label column
    set_label = list(set(label_column)) #identifying unique class
    unique_label_probability = []

    for unique_label in set_label:
        unique_label_probability.append(label_column.count(unique_label)/len(label_column)) #calculating probability of every label in the dataset

    def calculateEntropy(probs):
        return (-torch.sum(probs * torch.log2(probs)))  #calculating entropy

    k = (calculateEntropy(torch.tensor(unique_label_probability)))
    return k

#input : tensor, attribute_number
#output : int/float



def get_avg_info_of_attribute(tensor : torch.Tensor, attribute : int):
    '''Return avg_info of the given attribute'''
    # Extracting the attribute column
    curr_attr_tensor = tensor[:, attribute]

    # Extracting unique values in this attribute
    unique_attr_vals= torch.unique(curr_attr_tensor)

    # Extracting label column
    label_column = tensor[:, -1]

    # initialising the avg info variable/accumulator
    avg_info_of_attribute = 0
    
    # Calculating proportion and entropy of every unique value in the attribute to get avg info gain
    for unique_val in unique_attr_vals :
        
        # Creating mini-dataset where attribute == unique_val
        mini_dataset = tensor[(curr_attr_tensor == unique_val)]

        # Finding proportion of each unique attr wrt to full dataset
        proportion = len(mini_dataset) / len(tensor)

        # Finding entropy of the unique attr wrt mini dataset
        unique_val_entropy = get_entropy_of_dataset(mini_dataset)

        # finding average information gain for this unique attribute value and adding it to average information gain of the attribute
        avg_info_of_attribute += proportion * unique_val_entropy
    
    return avg_info_of_attribute


#input : tensor, attribute number
#output : int/float

def get_information_gain(tensor : torch.Tensor, attribute : int):
    '''Returns the information gain of the given attribute'''
    return (torch.round(torch.tensor(get_entropy_of_dataset(tensor)) - get_avg_info_of_attribute(tensor, attribute), decimals=4)).item()

#input : tensor
#output : ((dict), int)

def get_selected_attribute(tensor : torch.Tensor):
    '''Returns a tuple with the first element as a dictionary which has IG of all columns
    and second element as an  integer representing attribute number of selected attribute

    example = {{0 : 0.123, 1 : 0.768, 2 : 1.23}, 2}
    '''
    # To get selected attribute, have to get attribute with max information gain
    attribute_gains = {}

    # Step 1 : get all attributes -> tensor[0] = one record of the dataset w all columns, -1 to ignore 'label' column
    for attribute_number in range(len(tensor[0]) - 1) :
        attribute_gains[attribute_number] = get_information_gain(tensor, attribute_number)
    
    # Step 2 : selecting attribute with the highest gain from dictionary attribute_gains
    max_gain_val = max(attribute_gains.values())

    for attr in attribute_gains : 
        if attribute_gains[attr] == max_gain_val :
            print(attribute_gains, attr)
            return (attribute_gains, attr)

    return ({}, -1)