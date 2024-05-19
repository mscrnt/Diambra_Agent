#!/bin/bash

# Get the directory of the script
SCRIPT_DIR=$(dirname "$0")

# Restart PostgreSQL service
echo "Restarting PostgreSQL service..."
sudo service postgresql restart

# Switch to the postgres user and execute the database commands
echo "Resetting the database..."
sudo -u postgres psql << EOF
DROP DATABASE IF EXISTS optuna_db;
CREATE DATABASE optuna_db;
GRANT ALL PRIVILEGES ON DATABASE optuna_db TO optuna_user;
\q
EOF

echo "Database optuna_db has been reset."

# Deleting log directories
echo "Deleting log directories..."
sudo rm -rf "$SCRIPT_DIR/tensorboard_logs"
sudo rm -rf "$SCRIPT_DIR/optuna_logs"

echo "Log directories have been reset."

