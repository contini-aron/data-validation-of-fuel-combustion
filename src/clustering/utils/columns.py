def get_groupby():
    return [
        # "Experiment DOI",
        # "Exp SciExpeM ID",
        # "Chem Model",
        # "Chem Model ID",
        # "Experiment Type",
        # "Reactor",
        "Target",
        # "Fuels",
        # "Phi",
        # "Phi0",
        # "Phi1",
        # "Pressure",
        # "P0",
        # "P1",
        # "Temperature",
        # "T0",
        # "T1",
        # "Score",
        # "Error",
        # "d0L2",
        # "d1L2",
        # "d0Pe",
        # "d1Pe",
        # "shift"
    ]


def get_columns() -> list[str]:
    """
    :return: default column names used for clustering
    """
    return [
        # "Experiment DOI",
        # "Exp SciExpeM ID",
        # "Chem Model",
        # "Chem Model ID",
        # "Experiment Type",
        # "Reactor",
        # "Target",
        # "Fuels",
        # "Phi",
        "Phi0",
        "Phi1",
        # "Pressure",
        "Pressure0",
        "Pressure1",
        # "Temperature",
        "Temperature0",
        "Temperature1",
        "Score",
        "Error",
        "d0L2",
        "d1L2",
        "d0Pe",
        "d1Pe",
        "shift",
    ]
