from hrpe.data.load import load_minute_data, load_hh_data
from hrpe.features.deltas import calculate_deltas
from hrpe.features.transform import minute_data_to_hh_data
from hrpe.models.ets import ETS
from hrpe.submit.create_submission import create_submissions_file


data = load_minute_data("staplegrove")
hh_data = minute_data_to_hh_data(data)
hh_data = calculate_deltas(hh_data)

aug_ets = ETS(48)
aug_ets.fit(hh_data)

aug_hh = load_hh_data("staplegrove").query("type == 'august'")[["time", "value"]]
aug_hh.columns = ["time", "value_mean"]

aug_preds = aug_ets.predict(aug_hh)
create_submissions_file(aug_preds, "august", "ml")
