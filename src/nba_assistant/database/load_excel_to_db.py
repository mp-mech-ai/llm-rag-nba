import pandas as pd
from nba_assistant.database.schemas import Team, Player, Stats
import sqlite3
import logging
from pydantic import ValidationError

def parse_players(df: pd.DataFrame, db_path: str) -> None:
    df = df.iloc[:, :45]
    df.columns = list(Player.model_fields)
    
    table_name = "Players"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for _, row in df.iterrows():
        try:
            player = Player(**row.to_dict())
            # Get the player data as a dictionary
            player_dict = player.model_dump()

            # Extract column names and values
            columns = ', '.join(player_dict.keys())
            placeholders = ', '.join(['?'] * len(player_dict))
            values = list(player_dict.values())

            # Construct and execute the INSERT statement
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)
        except ValidationError:
            logging.warning(f"Skipping row {str(row).replace("\n", " ")[:40]} due to ValidationError")

    conn.commit()
    conn.close()
    nb_players = len(df)
    logging.info(f"{nb_players} players parsed")

def parse_teams(df: pd.DataFrame, db_path: str) -> None:
    df.columns = list(Team.model_fields)
    
    table_name = "Teams"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for _, row in df.iterrows():
        try:
            player = Team(**row.to_dict())
            # Get the player data as a dictionary
            player_dict = player.model_dump()

            # Extract column names and values
            columns = ', '.join(player_dict.keys())
            placeholders = ', '.join(['?'] * len(player_dict))
            values = list(player_dict.values())

            # Construct and execute the INSERT statement
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)
        except ValidationError:
            logging.warning(f"Skipping row {str(row).replace('\n', ' ')[:40]} due to ValidationError")

    conn.commit()
    conn.close()
    nb_teams = len(df)
    logging.info(f"{nb_teams} teams parsed")

def parse_stats(df: pd.DataFrame, db_path: str) -> None:
    df.columns = list(Stats.model_fields)
    
    df["stat_code"] = list(Player.model_fields)
    df.loc[df["stat_code"] == "pts", "stat_definition"] = "Points marqués totaux"
    df.loc[df["stat_code"] == "fgm", "stat_definition"] = "Tirs réussis totaux (Field Goals Made)"
    df.loc[df["stat_code"] == "fga", "stat_definition"] = "Tirs tentés totaux (Field Goals Attempted)"

    table_name = "Stats"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    for _, row in df.iterrows():
        try:
            player = Stats(**row.to_dict())
            # Get the player data as a dictionary
            player_dict = player.model_dump()

            # Extract column names and values
            columns = ', '.join(player_dict.keys())
            placeholders = ', '.join(['?'] * len(player_dict))
            values = list(player_dict.values())

            # Construct and execute the INSERT statement
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(sql, values)
        except ValidationError:
            logging.warning(f"Skipping row {str(row).replace('\n', ' ')[:40]} due to ValidationError")

    conn.commit()
    conn.close()
    nb_stats = len(df)
    logging.info(f"{nb_stats} stats parsed")

def load_xls_to_db(file_path, db_path, sheet_name_to_table):
    file = pd.ExcelFile(file_path)

    for sheet_name in file.sheet_names:
        if sheet_name in sheet_name_to_table.keys():
            logging.info(f"Traitement de la feuille {sheet_name}")

            df = file.parse(sheet_name)
            sheet_name_to_table[sheet_name](df, db_path)


if __name__=="__main__":
    from nba_assistant.utils.logging_handler import setup_logging
    import os
    setup_logging()

    sheet_name_to_table = {
        "Données NBA": parse_players,
        "Equipe": parse_teams,
        "Dictionnaire des données": parse_stats
    }

    from nba_assistant.config.config import DATABASE_FILE, INPUT_DIR

    load_xls_to_db(
        file_path=os.path.join(INPUT_DIR, "regular NBA.xlsx"),
        db_path=DATABASE_FILE, 
        sheet_name_to_table=sheet_name_to_table
        )