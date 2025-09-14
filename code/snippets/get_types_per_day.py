import pandas as pd


# Load metadata
types = pd.read_pickle('../../data/metadata_reduced.pickle')


# ----------------------- Inspection -----------------------
# Take a look at it
print("A first look at the data:")
print(types)


# Take a look at the most important columns (sorted by date)
date_type = types[["datetimes", "types"]].sort_values(by="datetimes")
print("\n The columns we are interested in:")
print(date_type)


# Show the datatypes of our relevant dataframe and statistical summary
print("\n The data types:")
print(date_type.info())
print("\n A statistical summary:")
print(date_type.describe(include="all"))
# -----------------------------------------------------------


# ----------------------- Main execution -----------------------
dates = date_type["datetimes"].dt.date.unique().tolist() # Generate a list of the available dates


results = {"Dates": [], "Types": []} # Initialize results' dictionary
results["Dates"] = dates # Include the available dates in the dictionary


# For each date, get the different kind of clouds and the number of them
for d in results["Dates"]:
    selected_dates = date_type[date_type["datetimes"].dt.date == d] # Select the images' types belonging to d day
    grouped = selected_dates.groupby("types", observed=False, dropna=False).size().reset_index(name="count") # Group by type of cloud and count 
    grouped = grouped[grouped["count"] != 0] # Remove type of clouds that are not present within the day
    results["Types"].append([t for t in grouped.itertuples(index=False, name=None)]) # Keep in types a list of tuples for each date


# Keep results in a dataframe (sorted by date, in case it got unsorted)
results_df = pd.DataFrame(results).sort_values(by="Dates")
print("\n", "-"*15, "The result:", "-"*15)
print(results_df)


# Export to .csv file:
#results_df.to_csv(path_or_buf="../../data/types_per_day_reduced.csv", index=False)

# Export to pickle file:
#results_df.to_pickle(path="types_per_day.pickle")
# ---------------------------------------------------------------