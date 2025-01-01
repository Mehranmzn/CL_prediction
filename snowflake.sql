-- Use an admin role
USE ROLE ACCOUNTADMIN;

-- Create the `transform` role
CREATE ROLE IF NOT EXISTS transform;
GRANT ROLE TRANSFORM TO ROLE ACCOUNTADMIN;

-- Create the default warehouse if necessary
CREATE WAREHOUSE IF NOT EXISTS CREDIT_APPLICATION;
GRANT OPERATE ON WAREHOUSE CREDIT_APPLICATION TO ROLE TRANSFORM;

-- Create the `dbt` user and assign to role
CREATE USER IF NOT EXISTS ds
  PASSWORD='dsPassword123'
  LOGIN_NAME='ds'
  MUST_CHANGE_PASSWORD=FALSE
  DEFAULT_WAREHOUSE='CREDIT_APPLICATION'
  DEFAULT_ROLE='transform'
  DEFAULT_NAMESPACE='ABN.RAW'
  COMMENT='DS user used for data transformation';
GRANT ROLE transform to USER ds;

-- Create our database and schemas
CREATE DATABASE IF NOT EXISTS ABN;
CREATE SCHEMA IF NOT EXISTS ABN.RAW;

-- Set up permissions to role `transform`
GRANT ALL ON WAREHOUSE CREDIT_APPLICATION TO ROLE transform; 
GRANT ALL ON DATABASE ABN to ROLE transform;
GRANT ALL ON ALL SCHEMAS IN DATABASE ABN to ROLE transform;
GRANT ALL ON FUTURE SCHEMAS IN DATABASE ABN to ROLE transform;
GRANT ALL ON ALL TABLES IN SCHEMA ABN.RAW to ROLE transform;
GRANT ALL ON FUTURE TABLES IN SCHEMA ABN.RAW to ROLE transform;






-- Set up the defaults
USE WAREHOUSE CREDIT_APPLICATION;
USE DATABASE ABN;
USE SCHEMA RAW;

-- Create the sales_ts table
CREATE OR REPLACE TABLE CREDIT_LOAN (
    client_nr INTEGER,
    yearmonth INTEGER,
    credit_application INTEGER,
    nr_credit_applications INTEGER,
    total_nr_trx INTEGER,
    nr_debit_trx INTEGER,
    volume_debit_trx INTEGER,
    nr_credit_trx INTEGER,
    volume_credit_trx INTEGER,
    min_balance INTEGER,
    max_balance INTEGER,
    CRG FLOAT
);





