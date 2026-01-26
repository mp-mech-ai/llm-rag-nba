import sqlite3
import logging
import os

def create_nba_database(db_name):
    if os.path.exists(db_name):
        raise FileExistsError(f"Database file {db_name} already exists. Please delete the file if you want to recreate it.")
    
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")

        # 1. Teams Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Teams (
            team_code TEXT PRIMARY KEY,
            team_name TEXT NOT NULL
        );
        """)

        # 2. Players Table (Identity + All Stats)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Players (
            player_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            team_code TEXT,
            age INTEGER,
            
            -- Performance Stats
            gp INTEGER, w INTEGER, l INTEGER, min REAL, pts INTEGER,
            fgm INTEGER, fga INTEGER, fg_pct REAL,
            three_pm INTEGER, three_pa INTEGER, three_p_pct REAL,
            ftm INTEGER, fta INTEGER, ft_pct REAL,
            oreb INTEGER, dreb INTEGER, reb INTEGER,
            ast INTEGER, tov INTEGER, stl INTEGER, blk INTEGER, pf INTEGER,
            fp REAL, dd2 INTEGER, td3 INTEGER, plus_minus REAL,
            
            -- Advanced Metrics
            off_rtg REAL, def_rtg REAL, net_rtg REAL,
            ast_pct REAL, ast_to REAL, ast_ratio REAL,
            oreb_pct REAL, dreb_pct REAL, reb_pct REAL,
            to_ratio REAL, efg_pct REAL, ts_pct REAL,
            usg_pct REAL, pace REAL, pie REAL, poss INTEGER,
            
            FOREIGN KEY (team_code) REFERENCES Teams(team_code)
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS Stats (
            stat_code TEXT PRIMARY KEY,
            stat_definition TEXT NOT NULL
        );
        """)

        conn.commit()
        conn.close()
        logging.info(f"Database '{db_name}' created successfully.")

    except sqlite3.Error as e:
        logging.error(f"Error: {e}")

# Run the function
if __name__ == "__main__":
    from nba_assistant.utils.logging_handler import setup_logging
    setup_logging()

    from nba_assistant.config.config import DATABASE_FILE
    create_nba_database(DATABASE_FILE)