import pickle

# Step 1: Open the file in read-binary mode
with open(r'src\fast\preprocessing\webscrape\lists_and_save_files\ordered_genres2.pkl', 'rb') as file:
    # Step 2: Load the serialized list
    loaded_genres = pickle.load(file)
    loaded_genres = [query.replace("/", "").lower()+" music" for query in loaded_genres]

# Step 3: Print the loaded list
print("Loaded list:")
print(len(loaded_genres))
# print(loaded_genres)
for q in loaded_genres:
    # if "epic" in q:
        # print(q)
    pass