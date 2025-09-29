from pipeline import CorePlacePipeline
from runs.curate.utils import save_dataconfig


def main():
    config = {
        "ReadGsvCities": {
            "gsv_cities_path": "/Users/olivergrainge/datasets/gsv-cities",
            "cities": ["Rome"],
        },
        "RandomClassSampler": {
            "num_classes": 50,
        },
        "Embeddings": {
            "model_name": "EigenPlaces",
            "batch_size": 32,
            "num_workers": 4,
        },
        "ComputeHardness": {
            "batch_size": 32,
            "margin": 0.1,
        },
        #"HardnessSampler": {
        #    "num_classes": 20,
        #    "num_instances_per_class": 3,
        #    "num_instances": None,
       #     "min_instances_per_class": 3,
       #     "percentile": None,
       #     "hardest_first": False,
       # },
        "RandomClassSampler": {
            "num_classes": 30,
            "num_instances_per_class": 3,
            "num_instances": None,
            "min_instances_per_class": 3,
        },

    }

    pipeline = CorePlacePipeline.from_config(config)
    pipe_state = pipeline.run()
    dataconfig = pipe_state["dataconfig"]
    save_dataconfig(dataconfig, "test")


if __name__ == "__main__":
    main()
