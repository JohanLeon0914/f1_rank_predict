from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UFC_CSV_DIR = PROJECT_ROOT / "UFC"
UFC_MODEL_DIR = PROJECT_ROOT / "models" / "ufc"
UFC_REPORT_DIR = PROJECT_ROOT / "reports" / "ufc"


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    required_columns: tuple[str, ...]


DATASET_SPECS = {
    "fighter": DatasetSpec(
        filename="fighter.csv",
        required_columns=(
            "fighter_id",
            "fighter_name",
            "height",
            "weight_lbs",
            "reach_inches",
            "stance",
            "dob",
            "slpm",
            "str_acc",
            "sapm",
            "str_def",
            "td_avg",
            "td_acc",
            "td_def",
            "sub_avg",
        ),
    ),
    "master": DatasetSpec(
        filename="master.csv",
        required_columns=(
            "fight_id",
            "event_id",
            "event_name",
            "event_date",
            "weight_class",
            "title_fight",
            "winner_id",
            "result_status",
            "method",
            "finish_round",
            "r_fighter_id",
            "b_fighter_id",
            "r_total_kd",
            "b_total_kd",
            "r_total_sig_landed",
            "r_total_sig_atmp",
            "b_total_sig_landed",
            "b_total_sig_atmp",
            "r_total_total_str_landed",
            "r_total_total_str_atmp",
            "b_total_total_str_landed",
            "b_total_total_str_atmp",
            "r_total_td_success",
            "r_total_td_atmp",
            "b_total_td_success",
            "b_total_td_atmp",
            "r_total_sub_att",
            "b_total_sub_att",
            "r_total_rev",
            "b_total_rev",
            "r_total_ctrl_seconds",
            "b_total_ctrl_seconds",
            "rounds_fought",
        ),
    ),
}

