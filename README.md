# BusinessAnalytics_RPIcase

This project trains 6 different models for bug triaging of the Eclipse dataset. For an overview of the task, check out the [RPI-Case file](docs/RPICase.pdf).

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

Download the data at https://github.com/ansymo/msr2013-bug_dataset. However, you need to convert either the xml files or the jsons into csv. Alternatively, you can drop me a pm to get the converted dataset. Store the tables in a new folder: data/Eclipse/

## Usage

Create the database and populate it:

```bash
cd db
sqlite3 Bugs.db
```
```bash
SQLite version 3.24.0 2018-06-04 14:10:15
Enter ".help" for usage hints.
sqlite> .read PopulateDB.sql
sqlite> .quit
```
```bash
cd ../src
```
#### For calibration
```bash
python Analysis.py -c
```
#### For calibration with exploratory analysis of features
```bash
python Analysis.py -c -p
```
#### For testing
```bash
python Analysis.py
```
## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/#)

