/*
	Import all the raw data from the Eclipse Bugzilla dataset.

	Author:	Florian Spychiger
	Date:	02 December 2018

	Stored in scripts/ImportData.sql
*/;

.mode csv

.import ../data/Eclipse/assigned_to.csv assigned_to
.import ../data/Eclipse/bug_status.csv bug_status
.import ../data/Eclipse/cc_cleaned.csv cc
.import ../data/Eclipse/component.csv component
.import ../data/Eclipse/op_sys.csv op_sys	
.import ../data/Eclipse/priority.csv priority	
.import ../data/Eclipse/product.csv product
.import ../data/Eclipse/reports.csv reports
.import ../data/Eclipse/resolution.csv resolution
.import ../data/Eclipse/severity.csv severity	
.import ../data/Eclipse/version.csv version


