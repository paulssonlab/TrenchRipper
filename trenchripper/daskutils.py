import pandas as pd
import numpy as np
import dask.dataframe as dd
import os
import shutil
import time

from .utils import writedir


def add_list_to_column(df, list_to_add, column_name, repartition=False):
    if repartition:
        df = df.repartition(partition_size="25MB").persist()

    index_name = df.index.name
    divisions = df.divisions

    df["index"] = 1
    idx = df["index"].cumsum()
    df["index"] = idx
    df = df.reset_index().set_index("index", sorted=True)

    list_to_add = pd.DataFrame(list_to_add)
    list_to_add["index"] = idx
    list_to_add = list_to_add.set_index("index")

    df = df.join(list_to_add, how="left", on="index")

    if divisions[0] != None:
        df = df.set_index(index_name, sorted=True, divisions=divisions)

    else:
        df = df.set_index(index_name)

    df.columns = df.columns.tolist()[:-1] + [column_name]

    return df


def set_new_aligned_index(df, index_column):
    ### Sets a column to be the new index

    ### Fast, but assumes index is both sorted and division aligned to the primary index

    first_indices = df.loc[list(df.divisions)].compute()
    new_index_divisions = first_indices[index_column].to_list()
    output_df = df.reset_index(drop=False).set_index(
        index_column,
        drop=True,
        sorted=True,
        npartitions=df.npartitions,
        divisions=new_index_divisions,
    )
    return output_df


def make_parquet_index(df, index_columns, index_precisions):
    index_precisions = index_precisions.astype("uint")
    rolling_precisions = np.add.accumulate(index_precisions[::-1])
    precision_factors = (10**rolling_precisions)[::-1]
    parquet_index = sum(
        [
            df[index_column].astype("uint") * precision_factors[i]
            for i, index_column in enumerate(index_columns)
        ]
    )
    return parquet_index


def lookup_parquet_index(index_lookup, index_precisions, index_columns):
    converted_index_lookup = np.array(
        [
            index_lookup[index_column] if index_column in index_lookup.keys() else -1
            for index_column in index_columns
        ]
    )
    converted_index_lookup_lower = converted_index_lookup.copy()
    converted_index_lookup_lower[converted_index_lookup_lower == -1] = 0
    converted_index_lookup_lower = converted_index_lookup_lower
    converted_index_lookup_upper = converted_index_lookup.copy()
    index_precisions_upper = index_precisions[:-1][
        converted_index_lookup_upper[1:] == -1
    ]
    converted_index_lookup_upper[converted_index_lookup_upper == -1] = np.array(
        [
            sum([9 * (10**i) for i in range(precision)])
            for precision in index_precisions_upper
        ]
    )

    index_precisions = index_precisions.astype("uint")
    rolling_precisions = np.add.accumulate(index_precisions[::-1])
    precision_factors = 10**rolling_precisions

    lower_parquet_index = sum(
        [
            index_column * precision_factors[i]
            for i, index_column in enumerate(converted_index_lookup_lower[::-1])
        ]
    ).astype("uint")
    upper_parquet_index = sum(
        [
            index_column * precision_factors[i]
            for i, index_column in enumerate(converted_index_lookup_upper[::-1])
        ]
    ).astype("uint")

    return lower_parquet_index, upper_parquet_index


def write_delayed_parition(delayed_partition, partition_idx, output_temp_path, engine):
    # helper function for to_parquet_checkpoint
    computed_partition = delayed_partition.compute()
    computed_partition.to_parquet(
        output_temp_path + "/temp_output." + str(partition_idx) + ".parquet",
        engine=engine,
    )
    with open(
        output_temp_path + "/Lockfile." + str(partition_idx) + ".txt", "w"
    ) as outfile:
        outfile.write("Done")
    return partition_idx


def to_parquet_checkpoint(
    dask_controller, dask_df, output_path, engine="pyarrow", overwrite=False
):
    # Performs a to_parquet write that is robust to crashes by first writing
    # a series of hdf5 dataframe while recording the status of this write in
    # metadata. Then it finally exports the final dataframe, cleaning up all
    # of the intermediate files. Can be used on any dask dataframe, in principle.
    # I have other code (in the main pipeline and in the growth quantification)
    # using a similar approach that should be converted later.

    output_temp_path = output_path + "_Temp"

    print(output_path)
    if overwrite:
        print("Starting Run.")
    else:
        if os.path.exists(output_path) and not os.path.exists(output_temp_path):
            print("Run Already Complete.")
            return None
        elif os.path.exists(output_temp_path):
            print("Resuming Run.")
        else:
            print("No Previous Run. Starting Run.")

    writedir(output_temp_path, overwrite=overwrite)

    # Check for existing files in case of in progress run
    if os.path.exists(output_temp_path):
        output_temp_path_dirlist = os.listdir(output_temp_path)
        finished_files = [
            int(path.split(".")[1])
            for path in output_temp_path_dirlist
            if "Lockfile" in path
        ]
    else:
        finished_files = []

    delayed_partition_list = dask_df.to_delayed()
    partition_list = list(range(len(delayed_partition_list)))
    unfinished_files = sorted(list(set(partition_list) - set(finished_files)))

    delayed_partition_list = [delayed_partition_list[i] for i in unfinished_files]
    output_futures = []

    for i, delayed_partition in enumerate(delayed_partition_list):
        partition_idx = unfinished_files[i]
        output_future = dask_controller.daskclient.submit(
            write_delayed_parition,
            delayed_partition,
            partition_idx,
            output_temp_path,
            engine,
        )
        output_futures.append(output_future)

    while len(unfinished_files) > 0:
        time.sleep(10)
        output_temp_path_dirlist = os.listdir(output_temp_path)
        finished_files = [
            int(path.split(".")[1])
            for path in output_temp_path_dirlist
            if "Lockfile" in path
        ]
        unfinished_files = sorted(list(set(partition_list) - set(finished_files)))

    temp_output_file_list = [
        output_temp_path + "/temp_output." + str(partition_idx) + ".parquet"
        for partition_idx in partition_list
    ]
    temp_df = dd.read_parquet(
        temp_output_file_list, engine=engine, calculate_divisions=True
    )

    dd.to_parquet(temp_df, output_path, engine=engine, overwrite=True)
    shutil.rmtree(output_temp_path)

    print("Done.")
