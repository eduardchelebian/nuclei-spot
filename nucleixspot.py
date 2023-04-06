import pandas as pd
import math
import seaborn as sns
from statannot import add_stat_annotation
import pandas as pd
import numpy as np



def nucleixspot(tissue_positions,
                 spot_diameter,
                 measurements,
                 position_coords,
                 measurements_coords=['Centroid X µm', 'Centroid Y µm'], 
                 qs=[0.5], 
                 selected_features=['Nucleus: Area'],
                 pixels=True,
                 resolution=1):

    """
    A function to aggregate the QuPath nuclei measurements from spots in a Visium spatial transcriptomics experiment.
    
    Args:
        tissue_positions: A pandas DataFrame containing the centroids of each Visium Spots.
        spot_diameter: A float representing the diameter of the spot (in pixels).
        measurements: A pandas DataFrame containing the QuPath detected Nuclei measurements.
        position_coords: A list of strings containing the column names of the x and y positions in the tissue_positions DataFrame.
        measurements_coords: A list of strings containing the column names of the x and y positions in the measurements DataFrame.
        qs: A list of quantiles to be calculated for each selected feature. Default is 0.5, the median.
        selected_features: A list of strings containing the names of the features to be aggregated. Default is ['Nucleus: Area'].
        pixels: A boolean representing wether the QuPath measurements are in pixels or microns
        resolution: A float represeting the image resolution in microns 
    
    Returns:
        A pandas DataFrame containing the aggregated measurements for each spot.
    """

    if not isinstance(spot_diameter, float):
        raise TypeError("Spot diameter should be a `float` retreived from scale_factors_json.spot_diameter_fullres")
    
    if not isinstance(tissue_positions, pd.DataFrame):
        raise TypeError("`tissue_positions` should be a DataFrame, received ", type(tissue_positions))

    if  not isinstance(measurements, pd.DataFrame):
        raise TypeError("`measurements` should be a DataFrame, received ", type(measurements))
    
    if not all(col in tissue_positions.columns for col in position_coords):
        raise ValueError("`position_coords` should be part of the `tissue_positions` DataFrame")

    if not all(col in measurements.columns for col in measurements_coords):
        raise ValueError("`measurements_coords` should be part of the `measurements` DataFrame")
    
    if not all(col in measurements.columns for col in selected_features):
        raise ValueError("`selected_features` should be part of the `measurements` DataFrame")

    qs = [qs] if isinstance(qs, (float, int)) else qs
    spot_radius = spot_diameter//2
    aggregated_measurements = pd.DataFrame()

    for i, row in tissue_positions.iterrows():
        # Use only the nuclei the centroid of which fall inside the spot
        x, y = row[position_coords[0]], row[position_coords[1]]
        mask = ((measurements[measurements_coords[0]] - x) ** 2 + (measurements[measurements_coords[1]] - y) ** 2 <= spot_radius ** 2)
        spot_measurements = measurements.loc[mask, selected_features]

        if not spot_measurements.empty:
            spot_aggregated = pd.DataFrame(index=[i])

            # Count number of detections 
            spot_aggregated['n_detections'] = int(len(spot_measurements))

            # Calculate the area percentage in each spot (spot_radius)
            if pixels:
                spot_aggregated['Nucleus: Area_percentage'] = (spot_measurements['Nucleus: Area'].sum() / (math.pi*(spot_radius**2))) * 100
            else: 
                spot_aggregated['Nucleus: Area_percentage'] = ((spot_measurements['Nucleus: Area'].sum()/resolution) / (math.pi*(spot_radius**2))) * 100

            # Calculate quantiles of the features
            for feature in spot_measurements.columns:
                for q in qs:
                    quantile_val = spot_measurements[feature].quantile(q)
                    new_col_name = f"{feature}_{int(q*100)}"
                    spot_aggregated[new_col_name] = [quantile_val]

            aggregated_measurements = pd.concat([aggregated_measurements, spot_aggregated])

    return aggregated_measurements



def nucleixspot_adata(adata, **kwargs):
    """
    Applies the `nucleixspot` function to adata.obs dataframe and adds the aggregated measurements as new columns.
    
    Parameters:
    -----------
    adata: AnnData object
        Annotated data matrix. Must contain an `obs` attribute which is a pandas DataFrame.
    **kwargs: keyword arguments
        Arguments to pass to the `nucleixspot` function.

    Returns:
    --------
    adata: AnnData object
        Annotated data matrix with aggregated measurements added to the `obs` attribute.
    """

    aggregated_measurements = nucleixspot(**kwargs)
    adata.obs = pd.concat([adata.obs, aggregated_measurements], axis=1)    
    adata.obs.iloc[:,-aggregated_measurements.shape[1]:] = adata.obs.iloc[:,-aggregated_measurements.shape[1]:].fillna(0)
    
    return adata


def morphology_differences(x, y, data, hue=None, 
                           box_pairs=None, stats=False, test='Mann-Whitney', 
                           text_format='star', loc='inside', verbose=2, 
                           stripplot=False, dodge=True, marker='.', 
                           alpha=0.5, color='black', jitter=0.4,
                           outliers=0,
                           **kwargs):

    """
    Plots a boxplot of y values against x values and hue, optionally
    with stripplot and statistical annotations.

    Parameters:
    -----------
    x: str
        Column name of x variable.
    y: str
        Column name of y variable.
    data: pandas.DataFrame
        Dataframe containing x and y variables.
    hue: str, optional
        Column name of hue variable.
    box_pairs: list, optional
        List of tuples containing pairs of box groups to compare.
    stats: bool, optional
        Flag to enable statistical annotations on the plot.
    test: str, optional
        Statistical test to use for comparisons. Default is 'Mann-Whitney'.
    text_format: str, optional
        Format of the p-value text. Default is 'star'.
    loc: str, optional
        Location of the text annotation. Default is 'inside'.
    verbose: int, optional
        Verbosity level of the statistical test. Default is 2.
    stripplot: bool, optional
        Flag to enable stripplot on the plot.
    dodge: bool, optional
        Flag to enable dodge on the plot.
    marker: str, optional
        Marker style for stripplot points. Default is '.'.
    alpha: float, optional
        Alpha level for the stripplot points. Default is 0.5.
    color: str, optional
        Color for the stripplot points. Default is 'black'.
    jitter: float, optional
        Amount of jitter to add to stripplot points. Default is 0.4.
    outliers: int, optional
        Number of standard deviations above the mean to consider outliers.
        If 0, outliers are not removed. Default is 2.
    kwargs: dict, optional
        Other keyword arguments to pass to seaborn functions.

    Returns:
    --------
    ax: seaborn.axisgrid.BoxPlot
        Axis object containing the plot.
    """
    
    if outliers:
        mean = data[y].mean()
        sd = data[y].std()
        data = data[data[y] <= mean + (outliers * sd)]

        percent_removed = (1 - (len(data) / len(data[y]))) * 100
        print(f'Removed {percent_removed:.2f}% of data as outliers')  


    ax = sns.boxplot(x=x, y=y, data=data, hue=hue, 
                     **kwargs)

    if stats:
        add_stat_annotation(ax, x=x, y=y, data=data, hue=hue, 
                            box_pairs=box_pairs, test=test, text_format=text_format, loc=loc, verbose=verbose,
                            **kwargs)

    if stripplot:
        sns.stripplot(x=x, y=y, data=data, hue=hue,
                      dodge=dodge, marker=marker, alpha=alpha, color=color, jitter=jitter,
                      **kwargs)

    return ax