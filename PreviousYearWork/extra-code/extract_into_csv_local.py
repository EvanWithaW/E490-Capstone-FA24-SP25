import pandas as pd
import csv


# https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html
# https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html
# https://www.geeksforgeeks.org/python-read-csv-using-pandas-read_csv/
# https://www.geeksforgeeks.org/writing-csv-files-in-python/#
# https://towardsdatascience.com/how-to-select-rows-from-pandas-dataframe-based-on-column-values-d3f5da421e93
# https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html


def create_csv_from_excel(excel_file, save_filename):
    # This has two sheets we need: sheet0 and sheet1
    dataframe0 = pd.DataFrame(pd.read_excel(excel_file, sheet_name=0))
    dataframe1 = pd.DataFrame(pd.read_excel(excel_file, sheet_name=1))

    frames = [dataframe0, dataframe1]
    write_df = pd.concat(frames)

    write_df.to_csv(save_filename,  header=True) 


excel_file = "ALPRPlateExportDaytime.xlsx"
data_location = "sheets/data.csv"
parsed_data = "sheets/parsed_data.csv"

# create_csv_from_excel(excel_file, data_location)

data_df = pd.read_csv("sheets/data.csv", usecols=["IMAGE1", "IMAGE2", "PLATE_READ", "PLATE_TYPE_CONFIDENCE", "PLATE_RDR_CONFIDENCE", "PLATE_TYPE"])

fields = ["PLATE_NUM", "IMAGE1", "IMAGE2", "PLATE_TYPE", "PLATE_TYPE_CONFIDENCE", "PLATE_RDR_CONFIDENCE"]

with open(parsed_data, "w", newline="", encoding="utf-8") as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    
    for i in range(len(data_df)):
        image_1 = data_df.loc[i]["IMAGE1"].split("/")[-1] if type(data_df.loc[i]["IMAGE1"]) != float else 0
        image_2 = data_df.loc[i]["IMAGE2"].split("/")[-1] if type(data_df.loc[i]["IMAGE2"]) != float else 0
        plate_num = data_df.loc[i]["PLATE_READ"]
        plate_type = data_df.loc[i]["PLATE_TYPE"]
        plate_type_conf = data_df.loc[i]["PLATE_TYPE_CONFIDENCE"]
        plate_rdr_conf = data_df.loc[i]["PLATE_RDR_CONFIDENCE"]

        row = [plate_num, image_1, image_2, plate_type, plate_type_conf, plate_rdr_conf]
        csvwriter.writerow(row)