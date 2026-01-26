from pydantic import BaseModel, Field
from typing import Optional

class Player(BaseModel):
    # Basic Info
    name: str = Field(..., description="Player name, required")
    team_code: Optional[str] = Field(default=None, description="Team code, foreign key", min_length=3, max_length=3)
    age: Optional[int] = Field(default=None, description="Player age")

    # Performance Stats
    gp: Optional[int] = Field(default=None, description="Games played")
    w: Optional[int] = Field(default=None, description="Wins")
    l: Optional[int] = Field(default=None, description="Losses")
    min: Optional[float] = Field(default=None, description="Minutes played")
    pts: Optional[int] = Field(default=None, description="Points")
    fgm: Optional[int] = Field(default=None, description="Field goals made")
    fga: Optional[int] = Field(default=None, description="Field goals attempted")
    fg_pct: Optional[float] = Field(default=None, description="Field goal percentage")
    three_pm: Optional[int] = Field(default=None, description="Three-pointers made")
    three_pa: Optional[int] = Field(default=None, description="Three-pointers attempted")
    three_p_pct: Optional[float] = Field(default=None, description="Three-point percentage")
    ftm: Optional[int] = Field(default=None, description="Free throws made")
    fta: Optional[int] = Field(default=None, description="Free throws attempted")
    ft_pct: Optional[float] = Field(default=None, description="Free throw percentage")
    oreb: Optional[int] = Field(default=None, description="Offensive rebounds")
    dreb: Optional[int] = Field(default=None, description="Defensive rebounds")
    reb: Optional[int] = Field(default=None, description="Total rebounds")
    ast: Optional[int] = Field(default=None, description="Assists")
    tov: Optional[int] = Field(default=None, description="Turnovers")
    stl: Optional[int] = Field(default=None, description="Steals")
    blk: Optional[int] = Field(default=None, description="Blocks")
    pf: Optional[int] = Field(default=None, description="Personal fouls")
    fp: Optional[float] = Field(default=None, description="Fantasy points")
    dd2: Optional[int] = Field(default=None, description="Double-doubles")
    td3: Optional[int] = Field(default=None, description="Triple-doubles")
    plus_minus: Optional[float] = Field(default=None, description="Plus-minus")

    # Advanced Metrics
    off_rtg: Optional[float] = Field(default=None, description="Offensive rating")
    def_rtg: Optional[float] = Field(default=None, description="Defensive rating")
    net_rtg: Optional[float] = Field(default=None, description="Net rating")
    ast_pct: Optional[float] = Field(default=None, description="Assist percentage")
    ast_to: Optional[float] = Field(default=None, description="Assist-to-turnover ratio")
    ast_ratio: Optional[float] = Field(default=None, description="Assist ratio")
    oreb_pct: Optional[float] = Field(default=None, description="Offensive rebound percentage")
    dreb_pct: Optional[float] = Field(default=None, description="Defensive rebound percentage")
    reb_pct: Optional[float] = Field(default=None, description="Total rebound percentage")
    to_ratio: Optional[float] = Field(default=None, description="Turnover ratio")
    efg_pct: Optional[float] = Field(default=None, description="Effective field goal percentage")
    ts_pct: Optional[float] = Field(default=None, description="True shooting percentage")
    usg_pct: Optional[float] = Field(default=None, description="Usage percentage")
    pace: Optional[float] = Field(default=None, description="Pace")
    pie: Optional[float] = Field(default=None, description="Player impact estimate")
    poss: Optional[int] = Field(default=None, description="Possessions")

class Team(BaseModel):
    team_code: str = Field(..., description="Team code, required", min_length=3, max_length=3)
    team_name: str = Field(..., description="Team name, required")

class Stats(BaseModel):
    stat_code: str = Field(..., description="Stat code, required")
    stat_definition: str = Field(..., description="Stat definition, required")


if __name__=="__main__":
    mapping = {}

    for i, field in enumerate(list(Player.model_fields)):
        mapping[i+1] = field
    
    print(mapping)