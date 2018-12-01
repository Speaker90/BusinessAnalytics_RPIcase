SELECT '[INFO] Importing raw data..';
.read ./scripts/ImportData.sql

SELECT '[INFO] Creating tables..';
.read ./scripts/CreateTables.sql

SELECT '[INFO] Populating tables..';
.read ./scripts/FillTables.sql

