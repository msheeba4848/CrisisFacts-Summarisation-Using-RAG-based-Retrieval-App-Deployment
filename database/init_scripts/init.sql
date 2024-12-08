-- create table structrure
CREATE TABLE IF NOT EXISTS all_data (
    id BIGINT,
    event TEXT,
    source TEXT,
    text TEXT,
    lang TEXT,
    lang_conf FLOAT,
    class_label TEXT,
    cleaned_text TEXT    
);

-- load CSV data
COPY all_data
FROM '/docker-entrypoint-initdb.d/all_data_cleaned.csv'
DELIMITER ',' 
CSV HEADER;

