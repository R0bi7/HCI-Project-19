import csv
# Age 176, Diagnosis Age 3

with open("dataset/combined_study_clinical_data.tsv") as f:
    rd = csv.reader(f, delimiter="\t", quotechar='"')
    first = True

    mergedTsvRows = list()
    for row in rd:
        if first:
            del row[3]
            mergedTsvRows.append(row)
            first = False
            continue

        age = row[176]
        diagnosisAge = row[3]

        if age == "NA" and diagnosisAge != "NA":
            row[176] = diagnosisAge
            del row[3]
            mergedTsvRows.append(row)
        elif age == "NA" and diagnosisAge == "NA":
            continue
        elif age != "NA":
            del row[3]
            mergedTsvRows.append(row)

    with open('changed_datasets/combined_study_clinical_data.tsv', 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        for row in mergedTsvRows:
            tsv_writer.writerow(row)
