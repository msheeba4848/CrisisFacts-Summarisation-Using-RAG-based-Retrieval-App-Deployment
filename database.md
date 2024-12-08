# **Database Documentation**

This document describes the methods, setup, and usage instructions for the database component of the Real-Time Crisis Summarization System.

---

## **Overview**

The database is implemented using **PostgreSQL** within a Docker container. It stores the cleaned and processed crisis-related data imported from a CSV file. This database is the foundation for data retrieval and further analysis.

---

## **Technologies Used**

- **PostgreSQL**: A powerful, open-source relational database system.  
- **Docker**: Used to containerize and manage the PostgreSQL database service.  
- **SQL**: Structured Query Language used for table creation and data import.  
- **CSV**: The format of the input dataset for storage into the database.

---

## **Files and Structure**

```plaintext
database/
│── Dockerfile.database       # Dockerfile to build the database container
│── init-scripts/
│   └── init.sql              # SQL script to create table and import data
data/
│── processed/
│   └── all_data_cleaned.csv  # Processed CSV file with cleaned data
```

---

## **Steps Implemented**

### **1. Table Creation**

The table `all_data` was created in the database to store specific fields extracted from the dataset:

```sql
CREATE TABLE IF NOT EXISTS all_data (
    id BIGINT,                -- Unique identifier (from CSV)
    event TEXT,               -- Event name
    source TEXT,              -- Data source
    text TEXT,                -- Original text
    lang TEXT,                -- Language code
    lang_conf FLOAT,          -- Language confidence score
    class_label TEXT,         -- Crisis classification label
    cleaned_text TEXT         -- Cleaned and preprocessed text
);
```

---

### **2. Data Import**

The dataset from `all_data_cleaned.csv` is loaded into the table using the `COPY` command:

```sql
COPY all_data(id, event, source, text, lang, lang_conf, class_label, cleaned_text)
FROM '/docker-entrypoint-initdb.d/all_data_cleaned.csv'
DELIMITER ','
CSV HEADER;
```

---

## **How to Use the Database**

### **1. Build the Database Docker Image**

Navigate to the `database/` directory and build the Docker image:

```bash
docker build -t database-image -f Dockerfile.database .
```

---

### **2. Run the Database Container**

Start the PostgreSQL container using the following command:

```bash
docker run -d --name database-container -p 5432:5432 database-image
```

- `-d`: Run the container in detached mode.  
- `--name`: Assign a name to the container.  
- `-p 5432:5432`: Map the PostgreSQL port (5432) to the host machine.

---

### **3. Connect to the Database**

Once the container is running, connect to the database using the `psql` command-line client or any PostgreSQL client tool.

#### **a) Using `psql` from the container**

1. Access the container:

   ```bash
   docker exec -it database-container bash
   ```

2. Connect to PostgreSQL:

   ```bash
   psql -U admin -d summarization_db
   ```

   - Username: `admin`  
   - Database: `summarization_db`

---

### **4. Query the Data**

Example SQL queries:

1. **Check total rows**:

   ```sql
   SELECT COUNT(*) FROM all_data;
   ```

2. **View first 10 rows**:

   ```sql
   SELECT * FROM all_data LIMIT 10;
   ```

3. **Filter data by `class_label`**:

   ```sql
   SELECT id, event, class_label, cleaned_text
   FROM all_data
   WHERE class_label = 'disaster_events';
   ```

---

## **Troubleshooting**

1. **Container already exists**:  
   Remove the existing container:

   ```bash
   docker rm -f database-container
   ```

2. **Data not loading**:  
   Ensure the `all_data_cleaned.csv` file is in the correct path (`data/processed/`).  
   Check logs:

   ```bash
   docker logs database-container
   ```

3. **Database initialization skipped**:  
   Delete existing volumes to force a fresh start:

   ```bash
   docker volume rm $(docker volume ls -q)
   ```

---

