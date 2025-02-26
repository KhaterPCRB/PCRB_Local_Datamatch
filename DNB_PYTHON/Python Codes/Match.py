#Test outside Jupyter...
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz, process
import numpy as np
from multiprocess import Pool
import string
tqdm.pandas()

##This portion just surpresses warnings that do not affect the performance of the code.
import warnings
warnings.filterwarnings("ignore")

#These Dictionaries ensure that the read csv command gets the correct data types for columns
O_col={'PRIMARYINSUREDNAME': str, 'STREETADDRESS_OPT': str, 'CITY_OPT': str, 'ZIP_OPT':str,
       'Cleaned_Name':str, 'OC_Add':str, 'O_Add_#':float, 'O_Add_Name':str}

G_col={'Business Name':str, 'Address 1':str, 'Address 2':str, 'City':str, 'State':str, 'Zip':str,
       'Geocoded Location':str, 'Trade name':str, 'Adj_BN':str, 'Adj_Add':str, 'G_Add_#': float,
       'G_Add_Name':str}
# #Dataread for Government Data
Gpath=r"C:\Users\khat\OneDrive - PENNSYLVANIA COMPENSATION RATING BUREAU\Desktop\DNB_Data_Matching\Local Data\Generated\GOV_PREPD.csv"
GDF=pd.read_csv(Gpath, dtype=G_col)
# #Dataread for Optimus
OP_path=r"C:\Users\khat\OneDrive - PENNSYLVANIA COMPENSATION RATING BUREAU\Desktop\DNB_Data_Matching\Local Data\Generated\OPTI_Unmatch_78.csv"
ODF=pd.read_csv(OP_path, dtype=O_col)
ODF['ZIP_OPT']= ODF['ZIP_OPT'].astype(str)
#Creating new columns for each dataset that just contains the first 4 digits zip code
#This allows for the code to only check nearby businesses
ODF['APPROX_ZIP_OPT'] = ODF['ZIP_OPT'].astype(str).str[:4].astype(int)
GDF['APPROX_ZIP_GOV'] = GDF['Zip'].astype(str).str[:4].astype(int)
#These are the submatching functions that actually apply the matching for individual rows
#================================================
#Secondary Address handler:
def suite_cleaner(text1, text2):
    ''' 
    Inputs: 
    text1: A single string of address data from optimus data
    text2: A single string of address data from Gov data

    Ouputs: A single Boolean

    Description: This function matches secondary address numbers/codes such as suite, floor, unit etc. and returns a boolean if the
    addresses have matching secondary address numbers. True indicates either a match or an assymetry of data, 
    meaning one or both of the addresses doesnt have a secondary address. This boolean is used in an upcoming function
    ''' 
    #Remove all spaces from both input strings
    text1 = text1.replace(" ", "")
    text2 = text2.replace(" ", "")

    #Find the generic pound sign symbolizing a the prescence of a secondary address
    index1 = text1.find('#')
    index2 = text2.find('#')
    
    #If either are missing the pound sign, return True
    if index1 == -1 or index2 == -1:
        return True 
    #Log the secondary address numbers
    else:
        start1 = index1 + 1
        start2 = index2 + 1    
        text1= text1[start1:]
        text2= text2[start2:]

        #Check for secondary Address equality
        if text1 == text2:
            return True
        else:
            return False


#==========================================
#Address Fuzzy Matcher:
def a_fm(num1,rem1,num2, rem2):
    '''
    Input:
    num1: The leading address number from an address string from optimus
    num2: The leading address number from an address string from Gov data
    rem1: The second part of an address string from optimus
    rem2: The second part of an address string from Gov

    Output:
    final_score: a match score from 0 - 1 that indicates the match score

    Description: This function using the simple rapidfuzz ratio (normalized distance) on the rem1 and rem2 part of the addresses, but
    also applies custom penalties based on address number and secondary address numbers. This ensures more accurate address matches.

    
    '''
    #Removes leading and trailing spaces
    
    rem1=str(rem1).strip()
    rem2=str(rem2).strip()   
    
    #Checks if leading numbers match
    num_match = num1 == num2 
    #Fuzzy match address info
    fuzz_score = fuzz.ratio(rem1, rem2) / 100
    
    #Check if secondary address number matches
    sc=suite_cleaner(rem1, rem2)
    
    #Apply penalties to fuzzy match score
    #If suite numbers mismatch, apply a -.2 penalty
    if sc is False:
        fuzz_score-=.2

    #If leading numbers do not match: apply a -.35 penalty
    if not num_match:
        fuzz_score -= 0.35  
    
    #Safety Catch to stop negative scores from occuring, thus setting the floor at 0
    final_score = max(fuzz_score, 0) 
    
    return final_score
#===========================================
#Name fuzzy match
def n_fm(b1,b2):
    ''' 
    Inputs: 
    b1: a single string containg the business name from optimus
    b2: a single string containg the business name from the government data

    Outputs:
    nscore: A fuzzymatch score from 0-1

    Description: This function averages two distinct fuzzy matching scoring algorithms, Weighted ratio and standard ratio,

    '''
    Rscore=fuzz.ratio(b1,b2)/ 100
    Wscore=fuzz.WRatio(b1,b2)/100
    nscore=(Wscore + Rscore)/2
   
    return(nscore)

#========================================================
#Full matching function:

def manual_extract_one(nquery, a_num, a_st, cquery, nchoices, addnums,addsts, cchoices):
    '''    
    Inputs: 
    nquery: a string containing a business name from Optimus
    a_num: a string containing a business address number from Optimus
    a_st: a string containing a business address remainder from Optimus
    cquery: a string containing a business' city from Optimus
    nchoices: a column of a dataframe containg strings of business names from Gov data
    addnums: a column of a dataframe containg strings of address numbers from Gov data
    addsts: a column of a dataframe containg strings of address remainders from Gov data
    cchoices: a column of a dataframe containg strings of business cities from Gov data

    Outputs:
    best_index: The index of the row in the Gov dataframe that is the best match for the current row in Optimus
    best_score: the best total Match Score for the row
    BestN: The name score for the best index row for business name
    BestA: The address score for the best index row for business address
    BestC: The city score for the best index row for business city

    Description: This function applies the name and address matching functions from above to whole rows of Governmen data. It selects a single row
    of Optimus data and then iterates row by row through the Gov data, until it finds the best match
    '''
    #Initializing best score and best index variables
    best_score = 0
    best_index = -1
    bestN=''
    bestA=''
    bestC=0
    Nscore=0
    Ascore=0
    Cscore=0
    #Main loop that goes row by row
    for index, (nchoice, a_num_choice, a_st_choice, cchoice) in enumerate(zip(nchoices, addnums, addsts, cchoices)):
        # Perform fuzzy matching for Name and Addresss
        Nscore = n_fm(nquery, nchoice)
        Ascore = a_fm(a_num, a_st, a_num_choice, a_st_choice)
        #City Score
        Cscore=n_fm(cquery, cchoice)
        #If city score is greater than .9, the penalty term is set to 0
        if Cscore >.9:
            Cmod=0
        #Apply a penalty term for low city score of -.1
        else:
            Cmod= .1
        # Average the scores and subtract the City penalty
        avg_score = ((0.5*Nscore) + (0.5*Ascore))-Cmod

        # Update best match if the new average score is better
        if avg_score > best_score:
            best_score = avg_score
            best_index = index
            bestN=Nscore
            bestA=Ascore
            bestC=Cscore
          
        # If a good match occurs, stop the loop to speed up processing time
        if best_score >= .92:
            break
  
    return best_index, best_score, bestN, bestA, bestC
#Function that applies manual_extract_one to every row of optimus
def find_match(row, choices):
    ''' 
    input:
    row: A single row of the Optimus dataframe
    choices: The government dataframe

    Output: A single list that contains all the outputs of the manual_extract_one. It will contain False and 0's in the event of an error

    Description: This function narrows down the potential choices of the Gov dataframe to only rows with the same first four digits of zipcode. It also handles errors and the event
    of an empty choices df
    '''
    try:
        #Filter down the potential matches by approx zip code
        choices_filtered = choices[(choices['APPROX_ZIP_GOV'] == row['APPROX_ZIP_OPT'])]

        if choices_filtered.empty:
            return [False, 0, 0, 0,0]
        
        # Call the manual_extract_one function for fuzzy matching scores
        index, total_score, namescore, addscore, cscore = manual_extract_one(row['Cleaned_Name'], 
                                                                             row['O_Add_#'], row['O_Add_Name'], 
                                                                             row['CITY_OPT'], choices_filtered['Adj_BN'], 
                                                                             choices_filtered['G_Add_#'], choices_filtered['G_Add_Name'], choices_filtered['City'])
        #Handle the case where no match is found
        if index is None:
            return [None, 0, 0, 0,0]
        # Return results when match is found
        else:
            original_index = choices_filtered.index[index]
         
            return [original_index, total_score, namescore, addscore, cscore]
    #Handle all errors and empty rows
    except Exception as e:
        # Return default values 
 
        print(f"Error in find_match: {e}")
        return [False, 0, 0, 0,0]
#================================================================================================
#Main Function
def Match(DF, Gov):
    ''' 
    input: 
    DF: This is the optimus df
    Gov: This is the Gov df

    Outputs:
    RDF: A formatted df with the matches appended

    Description: This function runs find match and handles loading bars, and reformats the results intoa df
    '''
    results = []  # List to collect result dictionaries
    errors=0
    # Iterate through rows of Optimus data
    #for _, row in tqdm(DF.iterrows(), total=DF.shape[0], desc='Performing Fuzzy Matching'): 
    for _, row in DF.iterrows(): 
        # Get the best match index and score from find_match
        MATCH = find_match(row, Gov)
    
        # Skip if no match found
        if not MATCH:
            continue

        # Unpack the match results into individual variables
        match_index, score, namescore, addscore, cscore = MATCH[0], MATCH[1], MATCH[2], MATCH[3], MATCH[4]
        
        # Check if a valid match was found
        if match_index is False:
            errors+=1
            # Handle case when no valid match is found by creating a blank row
            blank = pd.DataFrame(np.nan, columns=Gov.columns, index=[0])
            mrow = blank.iloc[0]
        else:
            # Get the matched row from Gov data using the index
            mrow = Gov.iloc[match_index]
        
        # Create a result dictionary with the selected data from both Optimus and Gov
        result = {
            "OPTI_Name": row['PRIMARYINSUREDNAME'],
            "Name Score": namescore,
            "Address Score": addscore,
            "City Score": cscore, 
            "Match Score": score,
            "Opti_Address": row['STREETADDRESS_OPT'], 
            "Opti_City": row['CITY_OPT'],  
            "Opti_Zip": row['ZIP_OPT']
        }

        # Add matched row data from Gov to result
        result.update(mrow.to_dict())  
        results.append(result)
    #Convert to a pandas dataframe
    RDF = pd.DataFrame(results)
    #Select custom columns
    RDF=RDF[['OPTI_Name','Business Name', 'Name Score',
              'Opti_Address', 'Address 1', 'Address 2', 'Address Score', 
              'Opti_City','City', "City Score",
              'Opti_Zip', 'Zip', 'Match Score',
              'State', 'Geocoded Location', 'Trade name', 'Adj_BN', 'Adj_Add', 'G_Add_#', 'G_Add_Name' ]]
    RDF=pd.DataFrame(RDF)
    print(f"{errors} errors occured")
    return RDF



#=============================================================
#SUB DF
df=ODF.sample(6000, random_state=1)

# #11 is a good seed
# R=main(df,GDF)


# #Resturcturing R


# RQ = R[(R['Match Score']>0.73) ] 
# RQ.sort_values(by='Match Score', ascending=True, inplace=True)
# print(f"{(len(RQ)/len(R))*100}% Match Rate")
# RQ.head(45)

def split_data(df, chunk_size):
    return [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
def run_para(OPTi_df, DNB_df, chunk_size=1000, threshold=0.73, verbose=False):
    chunks = split_data(OPTi_df, chunk_size)
    
    # Log the number of chunks for debugging purposes
    print(f"Number of chunks created: {len(chunks)}")

    # Use parallel processing
    with Pool(processes=4) as pool:
        results = pool.starmap(Match, [(chunk, DNB_df) for chunk in chunks])

    # Log each chunk length to make sure none are dropped
    for i, result in enumerate(results):
        print(f"Chunk {i} length: {len(result)}")

    # Concatenate all the results, ensuring no rows are dropped
    prefinal_results = pd.concat(results, ignore_index=True)
    
    final_results = prefinal_results[(prefinal_results['Match Score']>threshold) ] 
    final_results.sort_values(by='Match Score', ascending=True, inplace=True)
    
    print(f"{(len(final_results)/len(prefinal_results))*100}% Match Rate")


    if verbose:
        print(f"Total number of rows after concatenation: {len(prefinal_results)}")
        final_results.head(30)

    return final_results

# The main function that will run when the script is executed directly
if __name__ == '__main__':
    result_df = run_para(df, GDF,1000,.73, True)
    print(result_df)
