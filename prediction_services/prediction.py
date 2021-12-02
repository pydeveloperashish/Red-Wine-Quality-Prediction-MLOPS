import yaml
import numpy as np
import joblib
import json
import os

params_path = "params.yaml"
schema_path = os.path.join("prediction_services", "schema.json")


class NotInRange(Exception):
    def __init__(self, message="Values entered are not in Range"):
        self.message = message
        super().__init__(self.message)


class NotInColumn(Exception):
    def __init__(self, message="Not in Columns"):
        self.message = message
        super().__init__(self.message)


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)[0]
    print(prediction)

    try:
        if 3 <= prediction <= 8:
            return prediction
        else:
            raise NotInRange

    except  NotInRange:
        return "Unexpected Result"


def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInColumn

    def _validate_values(col, val):
        schema = get_schema()
        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]):
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col, val)
    return True


def form_response(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        print(response)
        return response


def api_response(dict_request):
    try:
        if validate_input(dict_request):
            data = np.array([list(dict_request.values())])
            response = predict(data)
            response = {"response": response}
            return response

    except NotInRange as e:
        response = {"the_expected_range_are": get_schema(), "response": str(e)}
        return response

    except NotInColumn as e:
        response = {"the_expected_columns are": list(get_schema().keys()), "response": str(e)}
        return response
