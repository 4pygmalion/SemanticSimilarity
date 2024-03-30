import os
import sys
import argparse
import numpy as np
from collections import OrderedDict

from tqdm import tqdm
from datetime import datetime
from itertools import islice
from pathlib import Path
from logging import Logger, Formatter, FileHandler, getLogger, DEBUG
from time import time
from typing import List, Literal, Dict
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig

SEMNATIC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SEMNATIC_DIR)
sys.path.append(ROOT_DIR)
from SemanticSimilarity.build_datasets import OntologyHandler
from SemanticSimilarity.data_model import Disease, Patient, PatientId
from SemanticSimilarity.calculator import (
    BaseSimilarityCalculator,
    NodeLevelSimilarityCalculator,
    FrequentBasedSimilarityCaculator,
    ResnikSimilarityCalculator,
    GeneDiseaseSimilarityCalculator,
    Phen2DiseaseCalculator,
)


class TopKListLenMissmatchException(Exception):
    pass


class SemanticSimilarityEvaluator:
    def __init__(self, logger: Logger, cases_logger: Logger) -> None:
        self.logger = logger
        self.cases_logger = cases_logger

    def draw_top_k_plot(
        self,
        top_ks: List[int],
        accuracies: List[float],
        file_path: Path,
    ) -> None:
        """
        top-k 값에 대한 정확도 그래프 그리는 메소드.

        Raises:
            TopKListLenMissmatchException: top-k 값과 accuracy 값이 1:1 대응되지
            않으면 예외 발생
        """
        if len(top_ks) != len(accuracies):
            raise TopKListLenMissmatchException

        plt.plot(top_ks, accuracies)
        plt.xlabel("top-k")
        plt.ylabel("accuracy")
        plt.title(file_path.name)
        plt.savefig(file_path)
        self.logger.info(f"top-k figure generated: {file_path}")

        plt.close("all")
        plt.clf()

    def get_top_k_recall(
        self,
        patients_map: Dict[str, Disease],
        predicted: Dict[str, Dict[str, Dict[str, float]]],
        top_k: int = 5,
    ) -> float:
        """
        1개의 질병을 가진 환자에 대한 top-k 정확도 계산

        Note:
            predicted는 정렬되어 있다고 가정.

        Args:
            patients_map (Dict[str, Disease]): 환자 질병-증상 정답 셋
            predicted (Dict[str, List[Tuple[str, float]]]): 예측 질병 리스트
            top_k (int, optional): top-k 값. Defaults to 5.

        Returns:
            float: top-k 정확도 반환

        Example:
            >>> patients_map = {
                    "ORPHA:231160": Disease(
                        disease_name="Familial cerebral saccular aneurysm",
                        disease_id="ORPHA:231160",
                        pheno={
                            Phenotype(pheno_name="Encephalomalacia", hpoid="HP:0040197"),
                            Phenotype(pheno_name="Aortic dissection", hpoid="HP:0002647"),
                            Phenotype(pheno_name="Intracranial hemorrhage", hpoid="HP:0002170"),
                            Phenotype(pheno_name="Aortic root aneurysm", hpoid="HP:0002616"),
                            Phenotype(pheno_name="Oculomotor nerve palsy", hpoid="HP:0012246"),
                        },
                    ),
                    ...
                    ...
                }
            >>> predicted = {
                    "OMIM:619340": {  # top-1 positive case
                            "OMIM:619340": 0.99,
                            "OMIM:619344": 0.58,
                            "OMIM:619345": 0.48,
                            "OMIM:619346": 0.38,
                            "OMIM:619347": 0.28,
                            "OMIM:619348": 0.18,
                            "OMIM:619349": 0.08,
                            "OMIM:619350": 0.88,
                            "OMIM:619341": 0.78,
                            "OMIM:619343": 0.68,
                        },
                    ...
                    ...
                }
            >>> top5_recall = self.get_top_k_recall(patients_map, predicted)
            >>> 0.98 #print(top5_recall)

        """
        corrects = 0
        incorrects = 0

        for patient_id, patient_obj in tqdm(
            patients_map.items(),
            desc="get recall::patients to go:",
            total=len(patients_map),
        ):
            if patient_obj.disease_id in list(predicted[patient_id].keys())[:top_k]:
                corrects += 1
            else:
                incorrects += 1
                self.cases_logger.info(f"{patient_id} NOT in top-{top_k} predictions")

        accuracy = corrects / (corrects + incorrects)
        self.logger.info(f"top-{top_k} accuracy is {accuracy:.4f}")
        return accuracy

    def get_top_k_recall_real_world(
        self,
        patients_map: Dict[PatientId, Patient],
        predicted: Dict[PatientId, Dict[str, Dict[str, float]]],
        top_k: int = 5,
    ) -> float:
        """
        n 개의 질병을 가진 환자에 대한 top-k 정확도 계산

        Note:
            predicted는 정렬되어 있다고 가정.

        Args:
            patients_map (Dict[PatientId, Patient]): 환자 질병-증상 정답 셋
            predicted (Dict[PatientId, Dict[str, Dict[str, float]]]): 예측 질병 리스트
            top_k (int, optional): top-k 값. Defaults to 5.

        Returns:
            float: top-k 정확도 반환

        Example:
            >>> patients_map = {
                    "EPJ21-PSFY": Patient(
                        id="EPJ21-PSFY",
                        diseases={
                            "OMIM:619340": Disease(
                                disease_id="OMIM:619340",
                                disease_name="DEVELOPMENTAL AND EPILEPTIC ENCEPHALOPATHY 96",
                                pheno={
                                    Phenotype(id="HP:0000559"),
                                    Phenotype(id="HP:0100689"),
                                    Phenotype(id="HP:0000615"),
                                    Phenotype(id="HP:0000613"),
                                    Phenotype(id="HP:0000518"),
                                },
                            ),
                            "OMIM:619341": Disease(
                                disease_id="OMIM:619341",
                                disease_name="PRAJA RING FINGER UBIQUITIN LIGASE 2",
                                pheno={Phenotype(id="HP:0006554")},
                            ),
                        },
                    ),
                    ...
                    ...
                }
            >>> predicted = {
                    "EPJ21-PSFY": {
                        "OMIM:619340": {
                            "OMIM:619340": 0.99,
                            "OMIM:619344": 0.88,
                            "OMIM:619345": 0.77,
                            "OMIM:619346": 0.66,
                            "OMIM:619347": 0.55,
                            "OMIM:619348": 0.44,
                            "OMIM:619349": 0.33,
                            "OMIM:619350": 0.22,
                            "OMIM:619341": 0.11,
                            "OMIM:619343": 0.01,
                        },
                        ...
                        ...
                }
            >>> top5_recall = self.get_top_k_recall_real_world(patients_map, predicted)
            >>> 0.98 #print(top5_recall)
        """
        corrects = 0
        incorrects = 0

        for patient_id, patient_obj in tqdm(
            patients_map.items(),
            desc="get recall:: patients to go: ",
            total=len(patients_map),
        ):
            for disease_id, patient_disease in patient_obj.diseases.items():
                top_k_pred = list(predicted[patient_id][disease_id].keys())[:top_k]
                if disease_id in top_k_pred:
                    corrects += 1
                else:
                    incorrects += 1
                    self.cases_logger.info(
                        f"{patient_id}:{disease_id} NOT in top-{top_k} predictions"
                    )
        accuracy = corrects / (corrects + incorrects)
        self.logger.info(f"top-{top_k} accuracy is {accuracy:.4f}")
        return accuracy


def run(
    args: argparse.Namespace,
    ontology_handler: OntologyHandler,
    testbed_runner: SemanticSimilarityEvaluator,
    calculator_config: DictConfig,
) -> None:
    top_ks = [1, 5, 10, 20, 30, 50, 100]
    diseases = None
    NUM_PATIENTS = args.n

    calculator = {
        "node_level": NodeLevelSimilarityCalculator,
        "resnik": ResnikSimilarityCalculator,
        "gene_disease": GeneDiseaseSimilarityCalculator,
        "freq_based": FrequentBasedSimilarityCaculator,
        "phen2disease": Phen2DiseaseCalculator,
    }
    LOG_FOLDER_NAME = "testbed_logs"

    # load disease from file
    if args.disease and os.path.isfile(args.disease):
        with open(args.disease, "r") as f:
            diseases = f.readlines()

    patients_json_path = Path(__file__).resolve().parent / LOG_FOLDER_NAME
    PATINENT_FILE_PATH = patients_json_path / f"{args.p}_{args.n}.json"
    REAL_PATIENT_ALL_PATH = patients_json_path / f"realworld_case_origin.json"

    if PATINENT_FILE_PATH.exists() and args.p != "real":  # load patients from json
        patients_map: Dict[str, Disease] = ontology_handler.load_from_json(PATINENT_FILE_PATH)
    elif REAL_PATIENT_ALL_PATH.exists() and args.p == "real":
        patients_map: Dict[PatientId, Patient] = ontology_handler.load_real_world_case(
            json_path=REAL_PATIENT_ALL_PATH, n_patients=args.n
        )
        print(f"----{args.n} real patients saved at {PATINENT_FILE_PATH}")
        ontology_handler.save_to_json(patients_map, PATINENT_FILE_PATH)
    else:
        patients_map: Dict[str, Disease] = (
            build_patients_from_scratch(
                ontology_handler=ontology_handler,
                build_option=args.p,
                diseases=diseases,
            )
            if args.f == None
            else ontology_handler.load_from_json(args.f)
        )

        print("----build patients from scratch")
        ontology_handler.save_to_json(patients_map, PATINENT_FILE_PATH)

    sliced_patients_map = dict(islice(patients_map.items(), args.n))
    AVG_PHENOS_PER_PATIENTS = (
        ontology_handler.get_avg_pheno_num(sliced_patients_map)
        if args.p != "real"
        else ontology_handler.get_avg_pheno_num_real_world(sliced_patients_map)
    )
    testbed_runner.logger.info(
        f"--average number of phenos in {args.n} {args.p} patients is: {AVG_PHENOS_PER_PATIENTS:.4f}"
    )
    for calculator_name, cal in calculator.items():
        calculator_instance = cal(config=calculator_config)
        calculator_instance.set_level()
        testbed_runner.logger.info("--nodel level is ready.")

        calculator_instance.set_mica_mat()
        testbed_runner.logger.info("--MICA matrix is reday.")
        calculator_instance.set_disease_pheno_map(
            disease_pheno_map=ontology_handler.disease_pheno_map
        )
        if calculator_name == "freq_based":
            calculator_instance.set_disease_hpo()
            testbed_runner.logger.info("--disease-pheno frequency is reday.")
        elif calculator_name == "gene_disease":
            calculator_instance.set_gene_info()
            testbed_runner.logger.info("--gene_info is reday.")

        testbed_runner.logger.info("testbed runner is good to go.")

        if args.p == "real":
            predicted: Dict[PatientId, Dict[str, Dict[str, float]]] = get_prediction_real_world(
                patients_map=sliced_patients_map, calculator=calculator_instance
            )
        else:
            predicted: Dict[str, Dict[str, float]] = get_prediction(
                patients_map=sliced_patients_map,
                calculator=calculator_instance,
            )

        accuracies = []
        testbed_runner.logger.info("-" * 80)
        testbed_runner.cases_logger.info("-" * 80)

        testbed_runner.logger.info(f"{calculator_name} result: ")
        testbed_runner.cases_logger.info(f"{calculator_name} result: ")

        top_k_recall_func = (
            testbed_runner.get_top_k_recall_real_world
            if args.p == "real"
            else testbed_runner.get_top_k_recall
        )
        for top_k in top_ks:
            accuracies.append(
                top_k_recall_func(
                    patients_map=sliced_patients_map,
                    predicted=predicted,
                    top_k=top_k,
                )
            )

        today = datetime.now().strftime("%y-%m-%d")
        figure_file_path = (
            Path(__file__).resolve().parent
            / LOG_FOLDER_NAME
            / f"{today}-{calculator_name}_{args.p}_{NUM_PATIENTS}_patients.png"
        )
        testbed_runner.draw_top_k_plot(
            top_ks=top_ks, accuracies=accuracies, file_path=figure_file_path
        )
        testbed_runner.logger.info(f"similarity method: {calculator_name} is done!")
        testbed_runner.cases_logger.info(f"similarity method: {calculator_name} is done!")


def get_prediction(
    patients_map: Dict[str, Disease],
    calculator: BaseSimilarityCalculator,
) -> Dict[str, Dict[str, float]]:
    """
    1 개 질환을 가진 환자들에 대해 주어진 calulator의 prediction 계산 후 정렬
    """
    predictions = dict()
    for patient_id, patient_obj in patients_map.items():
        outputs: Dict[str, float] = calculator.get_disease_similarty(patient_obj.pheno)
        predictions[patient_id] = dict(sorted(outputs.items(), key=lambda x: x[1], reverse=True))

    return predictions


def get_prediction_real_world(
    patients_map: Dict[PatientId, Patient],
    calculator: BaseSimilarityCalculator,
) -> Dict[PatientId, Dict[str, Dict[str, float]]]:
    """
    pred function for actual patients.

    Note:
        각 환자는 1개 이상의 질병을 가질 수 있음.
    Args:
        patients_map (Dict[PatientId, Patient]): dictionary map for actual patients
        calculator (BaseSimilarityCalculator): resnik, freq, gene etc..
    Examples:
        >>> get_prediction_real_world(actual_patients, calculator)
        >>>
    """

    predictions = dict()
    for patient_id, patient_obj in tqdm(
        patients_map.items(),
        total=len(patients_map),
    ):
        predictions[patient_id] = get_prediction(
            patients_map=patient_obj.diseases, calculator=calculator
        )
    return predictions


def build_patients_from_scratch(
    ontology_handler,
    build_option: Literal["optimal", "noisy", "ambiguous"],
    diseases: List[str] = None,
):
    build_methods = {
        "optimal": ontology_handler.build_optimal_patients_map,
        "noisy": ontology_handler.build_noisy_patients_map,
        "ambiguous": ontology_handler.build_ambiguous_patient_map,
    }

    START = time()
    patients_map = build_methods[build_option](diseases)
    logger.info(f"{build_option}_patients_map time: {time() - START:.5f} sec.")
    return patients_map


def set_logger(
    module_name: str = "testbed",
    logs_dir: Path = Path(__file__).resolve().parent,
):
    LOG_DIR = logs_dir / "testbed_logs"

    if not LOG_DIR.exists():
        os.mkdir(LOG_DIR)

    start_time = datetime.now().strftime("%Y-%m-%d")
    log_file_path = LOG_DIR / f"{module_name}-{start_time}.log"

    logger_formatter = Formatter(
        fmt="{asctime}\t{name}\t{filename}:{lineno}\t{levelname}\t{message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )

    file_handler = FileHandler(filename=log_file_path)
    file_handler.setFormatter(logger_formatter)
    file_handler.setLevel(DEBUG)

    logger = getLogger(module_name)
    logger.setLevel(DEBUG)
    logger.addHandler(file_handler)

    return logger


def args_parse():
    parser = argparse.ArgumentParser(description="Semantic Similarity Assessment")
    parser.add_argument(
        "--f",
        default=None,
        help="json file path to build dataset",
        required=False,
    )
    parser.add_argument(
        "--p",
        default="optimal",
        choices=["optimal", "noisy", "ambiguous", "real"],
        help="choose patients category",
        required=False,
    )
    parser.add_argument(
        "--n",
        default=10,
        type=int,
        help="number of patients",
        required=True,
    )
    parser.add_argument(
        "--disease",
        default=None,
        help="disease list path ",
        required=False,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = args_parse()
    logger = set_logger(module_name=f"testbed-{args.n}-{args.p}-patients")
    cases_logger = set_logger(module_name=f"false-cases-{args.n}-{args.p}-patients")
    config = OmegaConf.load(os.path.join(SEMNATIC_DIR, "config.yaml"))

    START = time()
    ontology_handler = OntologyHandler(config)
    logger.info(f"ontology_handeler build time: {time()-START} sec.")

    testbed_runner = SemanticSimilarityEvaluator(logger=logger, cases_logger=cases_logger)
    run(
        args=args,
        ontology_handler=ontology_handler,
        testbed_runner=testbed_runner,
        calculator_config=config,
    )
