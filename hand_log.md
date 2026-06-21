createdb lora_serve 
    create the Postgres database

        sends one SQL command to the running Postgres process

        /opt/homebrew/var/postgresql@16/base/
    1/          ← template1 (system template)
    16384/      ← postgres (default db)
    16385/      ← lora_serve (just created)

        Postgres doesn't create databases from scratch — it copies template1, a built-in template database that contains the system catalog tables, default schemas, built-in functions, and data types. Your new lora_serve database starts as an exact clone of template1.

        Postgres maintains a global system catalog called pg_database -- a table that tracks every database in the cluster
            CREATE DATABASE adds a row to pg_database with the new database's name, OID, owner, encoding, and other metadata
        
        postgresql://gagan@localhost/lora_serve tells the client which database to use

        Postgres process is still the same single process — it doesn't spawn a new process per database. 

