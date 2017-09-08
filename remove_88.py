import csv

input_csv = "20170816-112833-9469-SubmissionCsvFile"
delete_indices = []

with open(input_csv + ".csv", 'r') as f:
	reader = csv.reader(f, delimiter=',')

	for row in reader:
		if "_88" not in row[0]:
			with open(input_csv + "_fixed.csv", 'a') as f1:
				writer = csv.writer(f1, delimiter=',')
				writer.writerow(row)
