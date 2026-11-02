"""Serving: the features of a servable model, rebuilt for a place that has no site record.

A prediction cell has ERA5-Land, a soil-moisture climatology, gridded static layers and a calendar;
a training site had sensors, a merged daily tree and a record of its own. This package builds, for one
place, the daily frame the trainer read -- the same column names, units and daily statistics -- and then
runs the trainer's own feature engineering on it, so no feature is defined a second time.

One place:

* :mod:`src.serving.base_columns`   the columns a served daily frame carries, and where each comes from
* :mod:`src.serving.era5_daily`     ERA5-Land hourly -> those daily columns; the solar columns
* :mod:`src.serving.engineer`       the trainer's steps from a daily frame to a model matrix
* :mod:`src.serving.dependencies`   which base columns and how many days of history a model needs
* :mod:`src.serving.model_bundle`   load a FINAL model, refuse one that cannot be served, predict
* :mod:`src.serving.site_parity`    serve the training sites and compare with the training tree

The grid (a store on disk, :mod:`src.serving.store`):

* :mod:`src.serving.grid`           the served cells of the forest mask and their climatology
* :mod:`src.serving.era5_gee_daily` the daily columns computed in Earth Engine
* :mod:`src.serving.acquire_era5`   Stage A: those columns for every cell and day, into the store
* :mod:`src.serving.build_inputs`   Stage B: cells, solar columns (:mod:`src.serving.solar_grid`), LAI (:mod:`src.serving.lai`)
* :mod:`src.serving.statics`        Stage B: the static layers, read as the site extraction read them
* :mod:`src.serving.frames`         one cell's daily frame from the store
* :mod:`src.serving.grid_predict`   Stage C: plan, predict blocks of cells, assemble a NetCDF
"""
