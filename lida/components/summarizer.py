import json
import logging
from typing import Union
import pandas as pd
from lida.utils import clean_code_snippet, read_dataframe
from lida.datamodel import TextGenerationConfig
from llmx import TextGenerator
import dataprofiler as dp
import warnings

system_prompt = """
You are an experienced data analyst that can annotate datasets. Your instructions are as follows:
i) ALWAYS generate the name of the dataset and the dataset_description
ii) ALWAYS generate a field description.
iii.) ALWAYS generate a semantic_type (a single word) for each field given its values e.g. company, city, number, supplier, location, gender, longitude, latitude, url, ip address, zip code, email, etc
You must return an updated JSON dictionary without any preamble or explanation.
"""

logger = logging.getLogger("lida")


class Summarizer():
    def __init__(self) -> None:
        self.summary = None
        
        
    ##UPDATED START - for DATAQUERY - 26SEPT2024
    def profile_dataset(self, df: pd.DataFrame) -> dict:
        """Profile the dataset using DataProfiler"""
        profile = dp.Profiler(df)  # Profile the dataset
        report = profile.report()  # Get the profiling report

        properties_list = []
        for idx, col_profile in enumerate(report['data_stats']):
            properties = {
                "column": col_profile["column_name"],
                "dtype": col_profile["data_type"],
                "semantic_type": "",  # This will be enriched later with LLM
                "description": "",
                "samples": col_profile["samples"],
                "num_unique_values": col_profile["statistics"].get("unique_count", None),
                "min": col_profile["statistics"].get("min", None),
                "max": col_profile["statistics"].get("max", None),
                "std": col_profile["statistics"].get("stddev", None),
            }
            properties_list.append({"column": col_profile["column_name"], "properties": properties})

        return properties_list
    ##UPDATED END - for DATAQUERY - 26SEPT2024
    

    def enrich(self, base_summary: dict, text_gen: TextGenerator,
               textgen_config: TextGenerationConfig) -> dict:
        """Enrich the data summary with descriptions"""
        logger.info(f"Enriching the data summary with descriptions")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": f"""
        Annotate the dictionary below. Only return a JSON object.
        {base_summary}
        """},
        ]

        response = text_gen.generate(messages=messages, config=textgen_config)
        enriched_summary = base_summary
        try:
            json_string = clean_code_snippet(response.text[0]["content"])
            enriched_summary = json.loads(json_string)
        except json.decoder.JSONDecodeError:
            error_msg = f"The model did not return a valid JSON object while attempting to generate an enriched data summary. Consider using a default summary or a larger model with higher max token length. | {response.text[0]['content']}"
            logger.info(error_msg)
            print(response.text[0]["content"])
            raise ValueError(error_msg + "" + response.usage)
        return enriched_summary

    def summarize(
            self, data: Union[pd.DataFrame, str],
            text_gen: TextGenerator, file_name="", n_samples: int = 3,
            textgen_config=TextGenerationConfig(n=1),
            summary_method: str = "default", encoding: str = 'utf-8') -> dict:
        """Summarize data from a pandas DataFrame or a file location"""

        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            # modified to include encoding
            data = read_dataframe(data, encoding=encoding)

        # Use DataProfiler to get the properties of each column
        data_properties = self.profile_dataset(data)

        # default single stage summary construction
        base_summary = {
            "name": file_name,
            "file_name": file_name,
            "dataset_description": "",
            "fields": data_properties,
        }

        data_summary = base_summary

        if summary_method == "llm":
            # two-stage summarization with LLM enrichment
            data_summary = self.enrich(
                base_summary,
                text_gen=text_gen,
                textgen_config=textgen_config)
        elif summary_method == "columns":
            # no enrichment, only column names
            data_summary = {
                "name": file_name,
                "file_name": file_name,
                "dataset_description": ""
            }

        data_summary["field_names"] = data.columns.tolist()
        data_summary["file_name"] = file_name

        return data_summary
