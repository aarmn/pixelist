from enum import Enum


class ProcessingStatus(str, Enum):
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class ProcessingMode(str, Enum):
    NO_INTERMEDIATE = "no_intermediate"
    NO_INTERMEDIATE_SHOW_FINAL = "no_intermediate_show_final"
    WITH_INTERMEDIATE = "with_intermediate"
    WITH_INTERMEDIATE_SHOW_FINAL = "with_intermediate_show_final"
    WITH_INTERMEDIATE_SHOW_ALL = "with_intermediate_show_all"
