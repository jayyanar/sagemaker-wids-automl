"""SageMaker AutoPilot Helpers.

This package contains helper classes and functions that are used in the candidates definition notebook. (ensemble mode)
"""

import boto3
import json
import sagemaker
import os

from os.path import join
from urllib.parse import urlparse
from time import gmtime, strftime

from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.dataset_definition.inputs import S3Input


class AutoMLLocalEnsembleRunConfig:
    """
   AutoMLLocalEnsembleRunConfig represents common configurations and assists in the creation of resources needed
   to start an AutoGluon processing job from notebook.
   """

    TRIAL_HYPERPARAMS = ["num_bag_sets", "included_model_types", "presets", "auto_stack", "num_bag_folds",
                         "num_stack_levels", "refit_full", "set_best_to_refit_full", "save_bag_folds"]
    ENV_VARS_FILE_NAME = 'env_variables.json'

    PARAM_OVERRIDES_FILE_NAME = 'metaparameters.json'

    OPTIONAL_ENV_VARS = ["component_job_step_name", "enable_local_model_insights", "disable_model_insights",
                         "disable_feature_importance", "instance_type", "trial_id_suffix"]

    def __init__(self, test_artifacts_path, base_automl_job_config, local_automl_job_config):
        """
        Args:
        :param test_artifacts_path: The local path that auto-ml related artifacts were transferred
        :param base_automl_job_config: a dictionary that contains base AutoML job config which is generated from a
        :param local_automl_job_config: a dictionary that contains inputs/outputs path convention of local run

        """
        self.region = sagemaker.session.Session().boto_region_name
        self.test_artifacts_path = test_artifacts_path
        self.local_automl_job_config = local_automl_job_config
        self.base_automl_job_config = base_automl_job_config
        # the job name of an existing AutoML managed run
        self.automl_job_name = base_automl_job_config["automl_job_name"]
        # the base s3 path where the managed AutoML job stores the intermediates
        self.automl_output_s3_base_path = base_automl_job_config["automl_output_s3_base_path"]

        self.automl_job_preprocessed_data_path = os.path.join(self.automl_output_s3_base_path, "preprocessed-data")

        # Auto ML local job config
        self.local_automl_job_name = local_automl_job_config["local_automl_job_name"]
        self.local_automl_job_output_s3_base_path = local_automl_job_config["local_automl_job_output_s3_base_path"]
        self.local_automl_job_output_s3_output_path = \
            self.local_automl_job_output_s3_base_path + "/sagemaker-automl-candidates"
        self.trial_configs = []
        self.agt_params = []
        self.load_trial_configs(test_artifacts_path + '/trial_configs')
        self.s3_client = boto3.client('s3')

        # Keep track of selected trial config
        self.selected_trial_config = None

        self.selected_trial_config_idx = None

        self.bucket_name = None
        self.output_file_path = None
        self.env_variable_input_channel_path = None

    def filter_agt_params(self, environment):
        return {key: environment[key] for key in self.TRIAL_HYPERPARAMS if key in environment}

    def load_trial_configs(self, local_path):
        for filename in os.listdir(local_path):
            if filename.endswith(".json"):
                file_path = os.path.join(local_path, filename)
                with open(file_path, "r") as file:
                    trial_config = json.load(file)
                    self.trial_configs.append(trial_config)
                    self.agt_params.append(self.filter_agt_params(trial_config['Environment']))

    def update_trial(self, param_overrides):
        self.selected_trial_config = self.trial_configs[self.selected_trial_config_idx]
        selected_trial_env = self.selected_trial_config['Environment'].copy()
        selected_trial_env.update(param_overrides)
        selected_trial_env["enable_validation_split"] = "false"
        selected_trial_env["input_channel_mode"] = "File"

        # Filter out optional env vars
        selected_trial_env = {key: value for key, value in selected_trial_env.items()
                              if key not in self.OPTIONAL_ENV_VARS}
        self.store_env_vars_to_local_path(selected_trial_env)
        self.upload_env_vars_to_s3()

    def store_json_file(self, content, file_name):
        os.makedirs(self.test_artifacts_path, exist_ok=True)
        with open(join(self.test_artifacts_path, file_name), 'w') as f:
            json.dump(content, f, indent=4)

    def store_env_vars_to_local_path(self, selected_trial_env):
        for key, value in selected_trial_env.items():
            if isinstance(value, list):
                # make sure presets/model_types arrays are converted back to comma separated String
                selected_trial_env[key] = ', '.join(str(item) for item in value)
        self.store_json_file(selected_trial_env, self.ENV_VARS_FILE_NAME)

    def upload_env_vars_to_s3(self):
        parsed_url = urlparse(self.local_automl_job_output_s3_base_path)
        self.bucket_name = parsed_url.netloc
        self.output_file_path = parsed_url.path.lstrip('/')
        try:
            local_file_path = join(self.test_artifacts_path, self.ENV_VARS_FILE_NAME)
            self.s3_client.upload_file(local_file_path, self.bucket_name,
                                       os.path.join(self.output_file_path, self.ENV_VARS_FILE_NAME))
            self.env_variable_input_channel_path = f"s3://{self.bucket_name}/" \
                                                   f"{os.path.join(self.output_file_path, self.ENV_VARS_FILE_NAME)}"
        except Exception as e:
            print("An error occurred while uploading env variables to S3 for AGT trial run:", str(e))
        os.remove(local_file_path)

    def get_inference_image(self):
        return self.selected_trial_config['AGTInferenceImage']

    def display_candidate(self):
        with open(join(join(self.test_artifacts_path, self.PARAM_OVERRIDES_FILE_NAME)), 'r') as file:
            json_data = json.load(file)
            self.display_json(json_data)
            self.update_trial(json_data)

    def display_json(self, json):
        from IPython.display import display, HTML
        html_table = "<table>"
        for key, value in json.items():
            html_table += f"<tr><td>{key}</td><td>{value}</td></tr>"
        html_table += "</table>"
        display(HTML(html_table))

    def to_html_table(self):
        return """
        <table>
        <tr><th rowspan=2>Base AutoML Job</th><th>Job Name</th><td>{base_job_name}</td></tr>
        <tr><th>Base Output S3 Path</th><td>{base_output_path}</td></tr>
        <tr><th rowspan=5>Interactive Job</th><th>Job Name</th><td>{local_job_name}</td></tr>
        <tr><th>Base Output S3 Path</th><td>{local_job_base_path}</td></tr>
        </table>
        """.format(
            base_job_name=self.automl_job_name,
            base_output_path=self.automl_output_s3_base_path,
            local_job_name=self.local_automl_job_name,
            local_job_base_path=self.local_automl_job_output_s3_base_path
        )

    def display(self):
        from IPython.display import display, Markdown

        display(
            Markdown(
                "This notebook is initialized to use the following configuration: "
                + self.to_html_table()
            )
        )

    def pretty_print_json(self, json_dict):
        data = json_dict.copy()

        # Iterate over the parsed data and convert string values
        for key, value in data.items():
            if isinstance(value, str):
                if value.lower() == "true" or value.lower() == "false":
                    # Convert to JSON boolean
                    data[key] = json.loads(value.lower())
                elif "," in value or key in ["included_model_types", "presets"]:
                    # Convert to list
                    data[key] = [item.strip() for item in value.split(",")]
                else:
                    try:
                        # Convert to number
                        if "." in value:
                            data[key] = float(value)
                        else:
                            data[key] = int(value)
                    except ValueError:
                        # Value remains as a string if it doesn't represent a boolean, number, or list
                        pass
        return data

    def select_trial(self, trials_dropdown):
        self.selected_trial_config_idx = trials_dropdown - 1
        self.store_json_file(self.pretty_print_json(self.agt_params[trials_dropdown - 1]),
                             self.PARAM_OVERRIDES_FILE_NAME)
        self.display_json(self.agt_params[trials_dropdown - 1])

    @property
    def dropdown(self):
        from ipywidgets import widgets

        return widgets.Dropdown(
            options=list(range(1, len(self.agt_params)+1))
        )

    def prepare_processor_args(self):
        return {
            "network_config": sagemaker.network.NetworkConfig(
                encrypt_inter_container_traffic=self.selected_trial_config["NetworkConfig"][
                    "EnableInterContainerTrafficEncryption"],
                enable_network_isolation=self.selected_trial_config["NetworkConfig"]["EnableNetworkIsolation"]
            ),
            "role": self.selected_trial_config["RoleArn"],
            "image_uri": self.selected_trial_config["AppSpecification"]["ImageUri"],
            "instance_count": self.selected_trial_config["ProcessingResources"]["ClusterConfig"]["InstanceCount"],
            "instance_type": self.selected_trial_config["ProcessingResources"]["ClusterConfig"]["InstanceType"],
            "volume_size_in_gb": self.selected_trial_config["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"],
            "max_runtime_in_seconds": self.selected_trial_config["StoppingCondition"]["MaxRuntimeInSeconds"]
        }

    def prepare_processing_inputs(self):
        list_processing_inputs = []
        for processing_input in self.selected_trial_config["ProcessingInputs"]:
            if processing_input["InputName"] == "feature-specification":
                input_channel = ProcessingInput(
                    input_name=processing_input["InputName"],
                    s3_input=S3Input(
                        s3_uri=processing_input["S3Input"]["S3Uri"],
                        local_path=processing_input["S3Input"]["LocalPath"],
                        s3_data_type=processing_input["S3Input"]["S3DataType"],
                        s3_input_mode=processing_input["S3Input"]["S3InputMode"],
                        s3_data_distribution_type=processing_input["S3Input"]["S3DataDistributionType"]
                    )
                )
                list_processing_inputs.append(input_channel)

        list_processing_inputs.extend([
            ProcessingInput(
                input_name="input_data",
                s3_input=S3Input(
                    s3_uri=os.path.join(self.automl_job_preprocessed_data_path, "tuning_data"),
                    local_path="/opt/ml/processing/input/data",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated"
                )
            ),
            ProcessingInput(
                input_name="input_headers",
                s3_input=S3Input(
                    s3_uri=os.path.join(self.automl_job_preprocessed_data_path, "header"),
                    local_path="/opt/ml/processing/input/header",
                    s3_data_type="S3Prefix",
                    s3_input_mode="File",
                    s3_data_distribution_type="FullyReplicated"
                )
            ),
            ProcessingInput(
                input_name="env",
                s3_input=S3Input(
                    s3_uri=self.env_variable_input_channel_path,
                    local_path="/opt/ml/processing/input/notebook_env"
                )
            )]
        )
        return list_processing_inputs

    def prepare_processing_outputs(self):
        list_processing_outputs = []
        for processing_output in self.selected_trial_config["ProcessingOutputConfig"]["Outputs"]:
            if processing_output["OutputName"] in ["ds", "output"]:
                output_channel = ProcessingOutput(
                    output_name=processing_output["OutputName"],
                    destination=self.local_automl_job_output_s3_output_path if processing_output[
                                                                                   "OutputName"] == "output" else
                    self.local_automl_job_output_s3_base_path,
                    source=processing_output["S3Output"]["LocalPath"],
                    s3_upload_mode=processing_output["S3Output"]["S3UploadMode"]
                )
                list_processing_outputs.append(output_channel)
        return list_processing_outputs

    def prepare_model_args(self):
        res = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=self.output_file_path)
        model_dir_key = None
        for content in res['Contents']:
            if 'model.tar.gz' in content['Key']:
                model_dir_key = content['Key']
        if not model_dir_key:
            raise ValueError("no model data find in the directory: {}"
                             .format(f"s3://{self.bucket_name}/{self.output_file_path}"))
        return {
            "model_data": f"s3://{self.bucket_name}/{model_dir_key}",
            "image_uri": self.get_inference_image(),
            "role": self.selected_trial_config["RoleArn"],
            "enable_network_isolation": self.selected_trial_config["NetworkConfig"]["EnableNetworkIsolation"]
        }


def uid():
    """Returns an identifier that can be used when creating SageMaker entities like training jobs.
    Currently returns a formatted string representation of the current time"""
    return strftime("%d-%H-%M-%S", gmtime())
