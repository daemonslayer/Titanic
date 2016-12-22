.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

#################################################################################
# GLOBALS                                                                       #
#################################################################################

BUCKET = [OPTIONAL] your-bucket-for-syncing-data (do not include 's3://')

#################################################################################
# COMMANDS                                                                      #
#################################################################################

# open the requirements.txt file present in root.
# if everything mentioned there is not installed, run these COMMANDS:
### make requirements
### make app
# if installed, run this command:
### make app
requirements :
	pip install -q -r requirements.txt

titanic :
	python titanic/main.py

app : requirements
	titanic

visualizations :
	python visualizations/visualizations.py

clean:
	find . -name "*.pyc" -exec rm {} \;

lint:
	flake8 --exclude=lib/,bin/,docs/conf.py .

sync_data_to_s3:
	aws s3 sync data/ s3://$(BUCKET)/data/

sync_data_from_s3:
	aws s3 sync s3://$(BUCKET)/data/ data/

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################
